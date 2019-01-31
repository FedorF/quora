import torch.nn as nn

class FeedForward(nn.Module):

    def __init__(self, vocab_size, max_len, hidden_dim=128, num_classes=2, emb_dim=300):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.affine = nn.Linear(emb_dim*max_len, hidden_dim)
        self.relu = nn.ReLU()
        self.affine2 = nn.Linear(hidden_dim, num_classes)


    def forward(self, x):
        out = self.embedding(x)
        out = self.affine(out.view(out.size()[1], -1))
        out = self.relu(out)
        out = self.affine2(out)
        return out



class LSTM(nn.Module):

    def __init__(self, embeddings, hidden_dim=128, num_layers=1):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(embeddings)
        self.embedding.weight.requires_grad = False
        self.lstm = nn.LSTM(input_size=self.embedding.embedding_dim,
                            hidden_size=hidden_dim,
                            num_layers=num_layers,
                            dropout=0.4)

        self.affine = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        out = self.embedding(x)
        out, h_c = self.lstm(out)
        out = out[-1, :, :]
        out = self.affine(out)
        return out




