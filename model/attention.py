import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, embedding_dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(embedding_dim, embedding_dim)
        self.key = nn.Linear(embedding_dim, embedding_dim)
        self.value = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, x):
        queries = self.query(x)
        keys = self.key(x)
        values = self.value(x)
        scores = torch.bmm(queries,
                           keys.transpose(1, 2)) / torch.sqrt(torch.tensor(x.size(-1),
                                                                           dtype=torch.float32))
        attention_weights = torch.softmax(scores, dim=-1)
        attended_values = torch.bmm(attention_weights, values)
        return attended_values