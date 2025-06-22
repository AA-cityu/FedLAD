import torch
import torch.nn as nn

class DeepLog(nn.Module):
    def __init__(
        self,
        input_size=768,
        hidden_size=64,
        num_layers=2,
        num_classes=2,
        dropout=0.5,
        bidirectional=True
    ):
        super(DeepLog, self).__init__()

        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=bidirectional
        )
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_output_size, num_classes)

    def forward(self, x):
        assert x.dim() == 3, f"Expected input shape [B, T, 768], but got {x.shape}"
        lstm_out, (h_n, _) = self.lstm(x)  # h_n: [num_layers * num_directions, batch, hidden_size]
        if self.bidirectional:
            # Concatenate the last forward and backward hidden states
            last_hidden = torch.cat((h_n[-2], h_n[-1]), dim=1)  # [batch, hidden*2]
        else:
            last_hidden = h_n[-1]  # [batch, hidden]
        out = self.dropout(last_hidden)
        out = self.fc(out)
        return out
