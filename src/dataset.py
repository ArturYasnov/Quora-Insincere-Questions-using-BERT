import numpy as np
import torch
from torch.utils.data import DataLoader


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer):

        self.labels = [label for label in data["target"]]
        self.texts = [
            tokenizer(
                text,
                padding="max_length",
                max_length=64,
                truncation=True,
                return_tensors="pt",
            )
            for text in data["question_text"]
        ]

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.texts[idx]

    def __getitem__(self, idx):
        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)

        return batch_texts, batch_y
