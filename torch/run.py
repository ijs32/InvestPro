import torch, os, sys, torch.nn as nn, torch.optim as optim
from model import NLPModel
from dataset import StatementDataset
from random import shuffle
from datetime import datetime
from trainer import Trainer


def main(n_files=5000):
    """Main function to train the model"""

    with os.scandir("../data/training-data/company-statements_gz") as dir:
        files = [file.name for file in dir]
        shuffle(files)
        files = files[:n_files] # last 500 reserved for validation and testing.

    statement_train_dataset = StatementDataset(files, pre_trained=True)

    torch.save(statement_train_dataset.text_vob,
               f'./saved_vocab/text_vob.pt')
    
    timestamp = datetime.now().strftime('%Y%m%d')
    epochs = 35
    seed = 42

    model = NLPModel(statement_train_dataset.text_vob.vectors)
    print(model)

    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.002)
    loss_fn = nn.L1Loss()

    train_loader = statement_train_dataset.get_dataloader()

    # Train the model.
    model_trainer = Trainer(model, loss_fn, optimizer)
    model_trainer.set_loaders(train_loader)
    model_trainer.set_tensorboard(f"pytorch_{timestamp}")
    model_trainer.train(epochs, seed)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.exit("Wrong number of arguments given. Usage: `python run.py number_of_files`")
    n_files = int(sys.argv[1])
    main(n_files)
