import torch
import time


def progress_bar(origin, current, total, loss, time, epoch, epochs, bar_length=60):
    fraction = current / total

    arrow = int(fraction * bar_length - 1) * \
        '=' + \
        '=' if current == total else int(fraction * bar_length - 1) * '=' + '>'
    padding = int(bar_length - len(arrow)) * ' '

    ending = '\n' if current == total else '\r'
    time = f' -- Time: {time:.2f}s'
    print(
        f'Epoch: {epoch + 1}/{epochs} -- {origin} Progress: [{arrow}{padding}] {int(fraction*100)}% -- Loss: {loss:.4f}{time}', end=ending)


def train_one_epoch(train_loader, model, optimizer, loss_fn, device, epoch, epochs):
    """Train the model for one epoch, and return the average loss"""

    start_time = time.time()
    running_tloss = 0.0

    for i, tdata in enumerate(train_loader):
        text, label = tdata
        text = text.to(device).long()
        label = torch.tensor(label, dtype=torch.long).to(device).long()

        optimizer.zero_grad()
        outputs = model(text)

        loss = loss_fn(outputs, label)
        loss.backward()

        optimizer.step()

        running_tloss += loss
        if i % 10 == 0 or i == len(train_loader)-1:
            current_time = time.time()
            progress_bar("Train", i + 1, len(train_loader),
                         loss, (current_time - start_time), epoch, epochs)

    return running_tloss / len(train_loader)


def validate_one_epoch(valid_loader, model, loss_fn, device):
    """Run the model against validation data, and return the average loss and accuracy"""

    running_vloss = 0.0

    num_correct = 0
    for i, vdata in enumerate(valid_loader):
        v_text, v_label = vdata
        v_text = v_text.to(device).long()
        v_label = torch.tensor(v_label, dtype=torch.long).to(device).long()

        v_outputs = model(v_text)
        # v_outputs = F.softmax(v_outputs, dim=1)

        vloss = loss_fn(v_outputs, v_label)
        running_vloss += vloss

        _, predicted = torch.max(v_outputs, 1)

        # update the number of correct predictions
        num_correct += (predicted == v_label).sum().item()

    return (running_vloss / len(valid_loader)), num_correct
