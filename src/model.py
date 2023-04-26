import torch
import torch.nn as nn

class NLPModel(nn.Module):
    """
    ### NLP Model
    """

    def __init__(self, embeddings):

        super().__init__()
        self.text_embedding = nn.Embedding.from_pretrained(
            freeze=True, embeddings=embeddings)
        self.batch_norm = nn.BatchNorm1d(300)
        self.conv_stack = nn.Sequential(
            nn.Conv1d(300, 256, 7),
            nn.Sigmoid(),
            nn.MaxPool1d(10),
            nn.Dropout(0.10),

            nn.Conv1d(256, 256, 7),
            nn.Sigmoid(),
            nn.Dropout(0.10),
            nn.Conv1d(256, 128, 7),
            nn.Sigmoid(),
            nn.Dropout(0.10),
            nn.Conv1d(128, 128, 7),
            nn.Sigmoid(),
            nn.Dropout(0.10),
            nn.Conv1d(128, 128, 7),
            nn.Sigmoid(),
            nn.Dropout(0.10),
            nn.Conv1d(128, 64, 7),
            nn.Sigmoid(),
            nn.MaxPool1d(5),
            nn.Dropout(0.10),

            nn.Conv1d(64, 64, 7),
            nn.Sigmoid(),
            nn.AdaptiveMaxPool1d(1),
            nn.Dropout(0.10),
        )
        self.linear_stack = nn.Sequential(

            nn.Linear(64, 32),
            nn.Sigmoid(),
            nn.Linear(32, 16),
            nn.Sigmoid(),
            nn.Dropout(0.15),

            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, text):
        text_embed = self.text_embedding(text)
        normalized = self.batch_norm(text_embed.transpose(1, 2))
        x = self.conv_stack(normalized)
        x = x.flatten(start_dim=1)
        x = self.linear_stack(x)

        return x
