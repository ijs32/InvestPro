from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.optim as optim
from model import NLPModel
from dataset import StatementDataset
import sys, os, random
from datetime import datetime
from training_helper import train_one_epoch, validate_one_epoch

TIMESTAMP = datetime.now().strftime('%Y%m%d')


def main(statement_val_dataset, statement_train_dataset):
    """Main function to train the model"""
    EPOCHS = 35
    WRITER = SummaryWriter()
    DEVICE = torch.device('mps')
    BEST_VAL_ACC = 0.0

    model = NLPModel(len(statement_train_dataset.text_vob))
    model.to(DEVICE)
    print(model)

    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)
    loss_fn = nn.CrossEntropyLoss()

    train_loader = statement_train_dataset.get_dataloader()
    valid_loader = statement_val_dataset.get_dataloader()

    for epoch in range(EPOCHS):

        # Train the model
        model.train(True)
        avg_tloss = train_one_epoch(
            train_loader, model, optimizer, loss_fn, DEVICE, epoch, EPOCHS)

        tb_x = epoch * len(train_loader)
        WRITER.add_scalar('Loss/train', avg_tloss, tb_x)

        # Validate the model
        model.train(False)
        avg_vloss, num_correct = validate_one_epoch(
            valid_loader, model, loss_fn, DEVICE)

        val_acc = num_correct / len(statement_val_dataset)
        print(f'LOSS train {avg_tloss:.4f} valid {avg_vloss:.4f}')
        print(f'ACCURACY valid {val_acc:.4f}')

        WRITER.add_scalars('Training vs. Validation Loss',
                           {'Training': avg_tloss, 'Validation': avg_vloss},
                           epoch + 1)
        WRITER.add_scalars('Validation Accuracy', {
                           'Accuracy': val_acc}, epoch + 1)
        WRITER.flush()

        # Track best performance, and save the model's state
        if val_acc > BEST_VAL_ACC:
            model_path = f'./saved_models/model_{int(val_acc * 100)}-percent_{TIMESTAMP}.pt'
            torch.save(model.state_dict(), model_path)
            BEST_VAL_ACC = val_acc


if __name__ == '__main__':

    with os.scandir("../data/clean-company-statements") as dir:
        files = [file.name for file in dir]

    files = files[:10000]
    train_files = files[:9900]
    statement_train_dataset = StatementDataset(train_files)

    torch.save(statement_train_dataset.text_vob,
               f'./saved_vocab/text_vob{TIMESTAMP}.pt')

    val_files = files[9900:]
    statement_val_dataset = StatementDataset(
        val_files, statement_train_dataset.text_vob)

    main(statement_val_dataset, statement_train_dataset)
