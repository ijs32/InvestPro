import torch, gzip
import torchtext as tt
import torch.nn as nn
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torchtext.vocab import GloVe

class StatementDataset(Dataset):
    def __init__(self, file_list, text_vob=None):
        self.data = file_list
        
        if text_vob is not None:
            self.text_vob = text_vob
        else:
            self.get_vocab()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """Get a single item from the dataset."""

        with gzip.open(f"../data/training-data/company-statements_gz/{self.data[index]}", "rt") as f:
            text = f.read()
        label = float(self.data[index].split("__")[0])

        return text, label

    def get_vocab(self):
        """Build vocabularies for description and text name."""

        toker = tt.data.get_tokenizer('basic_english')
        file_names = self.data

        def yield_tokens(file_names):
            for file_name in file_names:
                with gzip.open(f"../data/training-data/company-statements_gz/{file_name}", "rt") as f:
                    yield toker(f.read())

        text_vob = tt.vocab.build_vocab_from_iterator(yield_tokens(
            file_names), min_freq=1, max_tokens=50000, specials=['<unk>', '<pad>'])
        text_vob.set_default_index(text_vob['<unk>'])

        self.text_vob = text_vob

    def collate_fn(self, batch):
        """Collate a batch of data into tensors."""
        padded_text = []
        labels = []
        for text, label in batch:
            num_text = torch.tensor([self.text_vob[word] or 0
                                    for word in text.split()])
            padded_text.append(num_text)

            labels.append(label)

        # pad text to match description

        padded_text[0] = nn.ConstantPad1d(
            (0, 128 - padded_text[0].shape[0]), 0)(padded_text[0])
        padded_text = pad_sequence(padded_text, batch_first=True)

        return padded_text, labels

    def get_dataloader(self):
        """Get a dataloader for the dataset."""
        return DataLoader(
            self, batch_size=4, shuffle=True, collate_fn=self.collate_fn)
