
import torch
import torch.nn as nn
from model import NLPModel
from dataset import StatementDataset
import sys, os
from training_helper import validate_one_epoch
from random import shuffle


def test(val_text_dataset):
    """Main function to train the model"""
    DEVICE = torch.device('mps')

    model = NLPModel(len(val_text_dataset.text_vob))
    model.load_state_dict(torch.load(
        './saved_models/model_0_20230326.pt'))
    model.eval()
    model.to(DEVICE)

    loss_fn = nn.MSELoss()

    valid_loader = val_text_dataset.get_dataloader()

    model.train(False)
    validate_one_epoch(valid_loader, model, loss_fn, DEVICE)


with os.scandir("../data/training-data/company-statements") as dir:
    files = [file.name for file in dir]
    shuffle(files)

files = files[:100]
text_vob = torch.load(f'./saved_vocab/text_vob.pt')
val_text_dataset = StatementDataset(
    files, text_vob)

test(val_text_dataset)
