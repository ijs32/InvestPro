import torch
import torch.nn as nn

class NLPModel(nn.Module):
    """
    ### NLP Model
    """

    def __init__(self, text_vocab_size):

        super().__init__()
        self.text_embedding = nn.Embedding(text_vocab_size, 128)
        self.batch_norm = nn.BatchNorm1d(128)
        self.conv_stack = nn.Sequential(
            nn.Conv1d(128, 256, 7),
            nn.ReLU(),
            nn.MaxPool1d(5),
            nn.Dropout(0.10),
            
            nn.Conv1d(256, 256, 7),
            nn.ReLU(),
            nn.MaxPool1d(5),
            nn.Dropout(0.10),

            nn.Conv1d(256, 256, 7),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),
            nn.Dropout(0.10),
        )
        self.linear_stack = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.15),

            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.15),

            nn.Linear(256, 321),
        )

    def forward(self, text):
        text_embed = self.text_embedding(text)
        normalized = self.batch_norm(text_embed.transpose(1, 2)).transpose(1, 2)
        x = self.conv_stack(normalized.transpose(1, 2)).transpose(1, 2)
        x = x.flatten(start_dim=1)
        logits = self.linear_stack(x)
        return logits
