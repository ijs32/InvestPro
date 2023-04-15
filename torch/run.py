import torch, os
import torch.nn as nn
import torch.optim as optim
from model import NLPModel
from dataset import StatementDataset
from random import shuffle
from datetime import datetime
from trainer import Trainer


def main(statement_train_dataset):
    """Main function to train the model"""

    timestamp = datetime.now().strftime('%Y%m%d')
    epochs = 35
    seed = 42

    model = NLPModel(statement_train_dataset.text_vob.vectors)
    print(model)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.002)
    loss_fn = nn.L1Loss()

    train_loader = statement_train_dataset.get_dataloader()

    model_trainer = Trainer(model, loss_fn, optimizer)
    model_trainer.set_loaders(train_loader)
    model_trainer.set_tensorboard(f"pytorch_{timestamp}")
    model_trainer.train(epochs, seed)

if __name__ == '__main__':
    with os.scandir("../data/training-data/company-statements_gz") as dir:
        files = [file.name for file in dir]
        files = files[:5000] # last 500 reserved for validation and testing.
        shuffle(files)

    statement_train_dataset = StatementDataset(files, pre_trained=True)

    torch.save(statement_train_dataset.text_vob,
               f'./saved_vocab/text_vob.pt')

    main(statement_train_dataset)
