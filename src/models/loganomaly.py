import torch
import torch.nn as nn
import torch.nn.functional as F

class LogAnomaly(nn.Module):
    def __init__(self, input_size=768, hidden_size=64, num_layers=1, num_keys=2, seq_len=28):
        super(LogAnomaly, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.sequence_length = seq_len

        # 双路 LSTM
        self.lstm0 = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.lstm1 = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        # Attention 参数
        self.attention_size = hidden_size
        self.w_omega = nn.Parameter(torch.zeros(hidden_size, hidden_size))
        self.u_omega = nn.Parameter(torch.zeros(hidden_size))

        self.fc = nn.Linear(2 * hidden_size, num_keys)

    def attention_net(self, lstm_output):
        # lstm_output: (batch, seq_len, hidden_size)
        batch_size, seq_len, _ = lstm_output.size()
        output_reshape = lstm_output.contiguous().view(-1, self.hidden_size)  # (batch*seq_len, hidden)
        attn_tanh = torch.tanh(torch.matmul(output_reshape, self.w_omega))  # (batch*seq_len, hidden)
        attn_scores = torch.matmul(attn_tanh, self.u_omega)  # (batch*seq_len,)
        attn_scores = attn_scores.view(batch_size, seq_len)  # ✅ 动态 reshape
        attn_weights = F.softmax(attn_scores, dim=1).unsqueeze(-1)  # (batch, seq_len, 1)
        attn_output = torch.sum(lstm_output * attn_weights, dim=1)  # (batch, hidden)
        return attn_output

    def forward(self, features):
        # 输入: features [batch, seq_len, input_size*2]
        input0, input1 = torch.chunk(features, 2, dim=-1)  # 各 [batch, seq_len, input_size]

        out0, _ = self.lstm0(input0)  # [batch, seq_len, hidden]
        out1, _ = self.lstm1(input1)  # [batch, seq_len, hidden]

        attn0 = self.attention_net(out0)  # [batch, hidden]
        attn1 = self.attention_net(out1)  # [batch, hidden]

        combined = torch.cat((attn0, attn1), dim=1)  # [batch, hidden*2]
        out = self.fc(combined)  # [batch, num_keys]
        return out
