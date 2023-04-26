import torch, os, sys, torch.nn as nn, torch.optim as optim
from model import NLPModel
from dataset import StatementDataset
from random import shuffle
from datetime import datetime
from trainer import Trainer
from loss import MeanSqrtError


def main(n_files=5000):
    """Main function to train the model"""

    with os.scandir("./data/company-statements") as dir:
        files = [file.name for file in dir]
        shuffle(files)
        t_files = files[:n_files] 
        v_files = files[100:] # last n reserved for validation and testing.

    statement_train_dataset = StatementDataset(t_files, pre_trained=True)
    statement_valid_dataset = StatementDataset(v_files, pre_trained=True)
    
    timestamp = datetime.now().strftime('%Y%m%d')
    epochs = 35
    seed = 42

    model = NLPModel(statement_train_dataset.text_vob.vectors)
    print(model)

    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.002)
    loss_fn = MeanSqrtError()

    train_loader = statement_train_dataset.get_dataloader()
    valid_loader = statement_valid_dataset.get_dataloader()

    # Train the model.
    model_trainer = Trainer(model, loss_fn, optimizer)
    model_trainer.set_loaders(train_loader, valid_loader)
    model_trainer.set_tensorboard(f"pytorch_{timestamp}")
    model_trainer.train(epochs, seed)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.exit("Wrong number of arguments given. Usage: `python run.py number_of_files`")
    n_files = int(sys.argv[1])
    main(n_files)
