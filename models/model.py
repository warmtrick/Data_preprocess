import torch.nn as nn

class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_filters, filter_sizes, max_length, num_classes, dropout=0.4):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels=embed_dim, out_channels=num_filters, kernel_size=h),
                nn.LeakyReLU(),
                nn.MaxPool1d(kernel_size=max_length-h+1)
            ) for h in filter_sizes
        ])
        self.fc = nn.Linear(num_filters * len(filter_sizes), num_classes)
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(0, 2, 1)
        x = [conv(x) for conv in self.convs]
        x = torch.cat(x, dim=1)
        x = x.view(-1, x.size(1))
        x = self.dropout(x)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x
