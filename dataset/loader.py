import os
import re
import string
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import OrderedDict
from torch.nn.utils.rnn import pad_sequence
from sklearn.utils import shuffle
from transformers import BertTokenizer, BertModel
from dataset.parser_registry import register_parser

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOKENIZER = BertTokenizer.from_pretrained('bert-base-uncased')
BERT_MODEL = BertModel.from_pretrained('bert-base-uncased', use_safetensors=False).to(DEVICE)


def sanitize(raw_text):
    raw_text = re.sub(r'[\[\]=,;()<>/-]', ' ', raw_text)
    raw_text = " ".join([tok.lower() if tok.isupper() else tok for tok in raw_text.strip().split()])
    raw_text = re.sub(r'([A-Z][a-z]+)', r' \1', re.sub(r'([A-Z]+)', r' \1', raw_text))
    raw_text = " ".join([w for w in raw_text.split() if not re.search(r'\d', w)])
    raw_text = raw_text.translate(str.maketrans('', '', string.punctuation))
    return " ".join([w.lower().strip() for w in raw_text.strip().split()])

def batch_encode(messages, device, skip_wordpiece=False):
    clean_msgs = []
    for text in messages:
        if skip_wordpiece:
            word_list = [w for w in text.split(" ") if w in TOKENIZER.vocab]
            text = " ".join(word_list)
        clean_msgs.append(text)

    inputs = TOKENIZER(clean_msgs, return_tensors='pt', padding=True, truncation=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = BERT_MODEL(**inputs)
        embs = torch.mean(outputs.last_hidden_state, dim=1)
    return embs.cpu()  # [batch_size, 768]


@register_parser("bgl")
@register_parser("thunderbird")
# --- register & parser BGL/Thunderbird ---
def parse_bgl_thunder(log_path, dev, win_len=1, stride=1, embed_type='bert', skip_wp=0):
    print(f"[INFO] Parsing file: {log_path}")
    with open(log_path, "r", encoding="latin1") as f:
        log_lines = [line.strip() for line in f]
    print(f"[INFO] Loaded {len(log_lines)} entries.")

    encoder_fn = batch_encode if embed_type.lower() == "bert" else None
    embedding_cache = {}
    data_seq, data_label = [], []

    for start_idx in tqdm(range(0, len(log_lines) - win_len, stride), desc="Encoding log windows"):
        log_window = log_lines[start_idx:start_idx + win_len]
        messages = []
        window_key = []

        # determine whether all the content in this window is normal or anomaly
        has_anomaly = 0 if all(line.startswith("-") for line in log_window) else 1

        for line in log_window:
            message = sanitize(line.split(" ", 1)[-1].lower())
            window_key.append(message)
            messages.append(message)

        # Find out which messages have not been encoded
        to_encode = [msg for msg in messages if msg not in embedding_cache]
        if to_encode:
            embs = encoder_fn(to_encode, dev, skip_wp)
            for msg, emb in zip(to_encode, embs):
                embedding_cache[msg] = emb

        # Assemble the window vector from the cache
        window_tensor = torch.stack([embedding_cache[m] for m in window_key])
        data_seq.append(window_tensor.detach().cpu().numpy())
        data_label.append(has_anomaly)

    print(f"[INFO] Shuffling and preparing arrays...")
    data_seq, data_label = shuffle(data_seq, data_label)
    print(f"[INFO] Done. Total sequences: {len(data_seq)} | Anomalies: {sum(data_label)}")

    return np.array(data_seq, dtype=np.float32), np.array(data_label, dtype=np.int32)

@register_parser("hdfs")
# --- register and parser HDFS ---
def parse_hdfs(log_path, label_path, dev, embed_type="bert", skip_wp=0):
    encoder_fn = batch_encode if embed_type.lower() == "bert" else None

    print(f"[INFO] Reading HDFS log file: {log_path}")
    with open(log_path, "r", encoding='utf8') as f:
        raw_lines = [line.strip() for line in f]

    print(f"[INFO] Reading label file: {label_path}")
    label_df = pd.read_csv(label_path)
    labels = label_df["Label"].tolist()

    assert len(raw_lines) == len(labels), "[ERROR] The number of lines in the log and the tags does not match!"

    embedding_cache = {}
    embedded_lines = []

    for line in tqdm(raw_lines, desc="Encoding HDFS log lines"):
        content = sanitize(line).lower()
        if content not in embedding_cache:
            emb = encoder_fn([content], dev, skip_wp)[0]  # shape: [dim]
            embedding_cache[content] = emb
        embedded_lines.append(embedding_cache[content])

    # reshape as [num_samples, 1, dim]
    embedded_tensor = torch.stack(embedded_lines).unsqueeze(1)

    print(f"[INFO] Done. Parsed {len(embedded_tensor)} log lines, {sum(labels)} anomalies.")
    return embedded_tensor.detach().cpu().numpy(), np.array(labels, dtype=np.int32)