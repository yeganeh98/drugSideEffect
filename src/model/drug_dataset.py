import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DrugGRU(nn.Module):
    def __init__(self, hidden_size, vocab_size, dropout_p=0.1):
        super(DrugGRU, self).__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.dropout_p = dropout_p

        self.embedding = nn.Embedding(self.vocab_size, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.decoder = nn.Linear(self.hidden_size, 1)

    def forward(self, x, target):
        # x is a tensor of size (1, num_words)
        _, num_words = x.size()
        hidden = self.initHidden()
        for i in range(num_words):
            embedded = self.embedding(x[:, i]).view(1, 1, -1)
            embedded = self.dropout(embedded)

            output, hidden = self.gru(embedded, hidden)
        logits = torch.sigmoid(self.decoder(hidden)).view(1, 1)

        # compute loss
        loss = F.binary_cross_entropy(logits, target)

        return loss

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)
