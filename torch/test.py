
import torch
import torch.nn as nn
from model import NLPModel
from dataset import PremierDataset
import sys
from training_helper import validate_one_epoch


def train(premier_val_dataset):
    """Main function to train the model"""
    DEVICE = torch.device('mps')

    model = NLPModel(len(desc_vob), len(vendor_vob))
    model.load_state_dict(torch.load(
        './saved_models/model_96-percent_20230304.pt'))
    model.eval()
    model.to(DEVICE)

    loss_fn = nn.CrossEntropyLoss()

    valid_loader = premier_val_dataset.get_dataloader()

    model.train(False)
    avg_vloss, num_correct = validate_one_epoch(
        valid_loader, model, loss_fn, DEVICE)

    val_acc = num_correct / len(premier_val_dataset)
    print(f'LOSS valid {avg_vloss:.4f}')
    print(f'ACCURACY valid {val_acc:.4f}')


if len(sys.argv) < 2:
    print("Usage: python main.py large | small")
    sys.exit(1)
else:
    file_size = sys.argv[1]
    print("Using file size: ", file_size)

desc_vob = torch.load(f'./saved_vocab/large_desc_vob.pt')

vendor_vob = torch.load(f'./saved_vocab/large_vendor_vob.pt')

test_csv_path = f'../data/{file_size}_premier_test.csv'
premier_val_dataset = PremierDataset(
    test_csv_path, desc_vob, vendor_vob)

train(premier_val_dataset)
