from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.optim as optim
from model import NLPModel
from dataset import StatementDataset
import sys, os
from random import shuffle
from datetime import datetime
from training_helper import train_one_epoch, validate_one_epoch

TIMESTAMP = datetime.now().strftime('%Y%m%d')


def main(statement_train_dataset):
    """Main function to train the model"""
    EPOCHS = 35
    WRITER = SummaryWriter()
    DEVICE = torch.device('mps')

    model = NLPModel(len(statement_train_dataset.text_vob))
    model.to(DEVICE)
    print(model)

    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.002)
    loss_fn = nn.MSELoss()

    train_loader = statement_train_dataset.get_dataloader()

    for epoch in range(EPOCHS):

        # Train the model
        model.train(True)
        avg_tloss = train_one_epoch(
            train_loader, model, optimizer, loss_fn, DEVICE, epoch, EPOCHS)

        tb_x = epoch * len(train_loader)
        WRITER.add_scalar('Loss/train', avg_tloss, tb_x)
        WRITER.flush()

        model_path = f'./saved_models/model_{epoch}_{TIMESTAMP}.pt'
        torch.save(model.state_dict(), model_path)


if __name__ == '__main__':

    with os.scandir("../data/training-data/company-statements_gz") as dir:
        files = [file.name for file in dir]
        shuffle(files)

    files = files[:3000]
    statement_train_dataset = StatementDataset(files)

    torch.save(statement_train_dataset.text_vob,
               f'./saved_vocab/text_vob.pt')

    main(statement_train_dataset)
