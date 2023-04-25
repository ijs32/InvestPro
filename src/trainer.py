import torch, datetime, numpy as np
from torch.utils.tensorboard import SummaryWriter


class Trainer(object):
    def __init__(self, model, loss_fn, optimizer):
        self.loss_fn = loss_fn
        self.optimizer = optimizer

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = model
        self.model.to(self.device)

        self.train_loader = None
        self.val_loader = None
        self.writer = None

        self.losses = []
        self.val_losses = []
        self.n_epochs = 0
        self.completed_epochs = 0

        self.train_step_fn = self._make_train_step_fn()
        self.val_step_fn = self._make_val_step_fn()

    def to(self, device):
      """change model to different device"""
      try:
          self.device = device
          self.model.to(self.device)
      except RuntimeError:
          self.device = ('cuda' if torch.cuda.is_available()
                          else 'cpu')
          print(f"Couldn't send it to {device}, \
                  sending it to {self.device} instead.")
          self.model.to(self.device)

    def set_loaders(self, train_loader, val_loader=None):
        """Set Class train and validation loaders."""
        self.train_loader = train_loader
        self.val_loader = val_loader

    def set_tensorboard(self, name, folder='runs'):
        """Create a tensorboard writer object."""
        suffix = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        self.writer = SummaryWriter(f'{folder}/{name}_{suffix}')
    
    def _make_train_step_fn(self):
        """Perform a training step"""
        def perform_train_step_fn(x, y):
          # Sets model to TRAIN mode
          self.model.train()

          yhat = self.model(x)
          loss = self.loss_fn(yhat, y)

          loss.backward()

          self.optimizer.step()
          self.optimizer.zero_grad()

          return loss.item()

        return perform_train_step_fn
    
    def _make_val_step_fn(self):
        """Returns function `perform_val_step_fn`"""
        def perform_val_step_fn(x, y):
            # Sets model to EVAL mode
            self.model.eval()

            yhat = self.model(x)
            loss = self.loss_fn(yhat, y)

            return loss.item()

        return perform_val_step_fn
    
    def _mini_batch(self, validation=False):
        """
        Function for performing a step.

        Defaults to training dataloader 
        
        passing `true` as the second arg will use validation loader
        """
        if validation:
            data_loader = self.val_loader
            step_fn = self.val_step_fn
            origin = "validation"
        else:
            data_loader = self.train_loader
            step_fn = self.train_step_fn
            origin = "train"

        if data_loader is None: return None

        mini_batch_losses = []
        for i, data in enumerate(data_loader):
            text, targets = data
            text = text.to(self.device).long()
            targets = (torch.tensor(targets, dtype=torch.float).to(self.device)).unsqueeze(1)

            mini_batch_loss = step_fn(text, targets)
            mini_batch_losses.append(mini_batch_loss)

            if i % 10 == 0 or i == len(data_loader)-1:
                self.progress_bar(origin, i + 1, len(data_loader),
                            mini_batch_loss)
        
        return np.mean(mini_batch_losses)
    
    def _set_seed(self, seed):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    def train(self, n_epochs, seed):
        """Train model for epochs = n_epochs."""
        self._set_seed(seed)
        self.n_epochs = n_epochs
        best_loss = 1_000_000
        for epoch in range(self.n_epochs):
            # Train the model
            loss = self._mini_batch(validation=False)
            self.losses.append(loss)
            # make sure we are using a validation set, else skip.
            if self.val_loader:
                with torch.no_grad():
                    val_loss = self._mini_batch(validation=True)
                    self.val_losses.append(val_loss)
            
            self.completed_epochs += 1
            if loss < best_loss:
                best_loss = loss
                model_path = f'./saved_models/model_{epoch}.pt'
                self.save_checkpoint(model_path)

        if self.writer:
            scalars = {'training': loss}
            if val_loss is not None:
                scalars.update({'validation': val_loss})

            self.writer.add_scalars(main_tag='loss',
                                    tag_scalar_dict=scalars,
                                    global_step=epoch)

        if self.writer:
            # Flushes the writer
            self.writer.flush()
    
    def save_checkpoint(self, filename):
        # Builds dictionary with all elements for resuming training
        checkpoint = {
            'epoch': self.completed_epochs,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss_fn': self.loss_fn(),
            'loss': self.losses,
            'val_loss': self.val_losses
        }
        torch.save(checkpoint, filename)

    def load_checkpoint(self, filename, continue_train=True):
        # Loads dictionary
        checkpoint = torch.load(filename)

        # Restore state for model and optimizer
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(
            checkpoint['optimizer_state_dict']
        )
        self.loss_fn = checkpoint['loss_fn']

        if continue_train:
            self.completed_epochs = checkpoint['epoch']
            self.losses = checkpoint['loss']
            self.val_losses = checkpoint['val_loss']

            self.model.train() # always use TRAIN for resuming training
        else:
            self.model.eval()
            
    def progress_bar(self, origin, current, total, loss, bar_length=60):
        """Show training progress with progress bar and stats."""
        fraction = current / total

        arrow = int(fraction * bar_length - 1) * \
            '=' + \
            '=' if current == total else int(fraction * bar_length - 1) * '=' + '>'
        padding = int(bar_length - len(arrow)) * ' '

        ending = '\n' if current == total else '\r'
        print(
            f'Epoch: {self.completed_epochs + 1}/{self.n_epochs} -- {origin} Progress: [{arrow}{padding}] {int(fraction*100)}% -- Loss: {loss:.4f}', end=ending)