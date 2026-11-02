import os
import urllib.request

import torch

DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "input.txt")


def load_text(path=DATA_PATH):
    if not os.path.exists(path):
        print("input.txt not found, downloading tiny shakespeare...")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        urllib.request.urlretrieve(DATA_URL, path)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


class CharData:
    """Character-level vocab plus a 90/10 train/val split.

    The split is just the first 90% of the file vs the last 10%, so the
    validation text is from plays the models never see during training.
    """

    def __init__(self, text, val_frac=0.1):
        self.chars = sorted(set(text))
        self.vocab_size = len(self.chars)
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.itos = {i: c for i, c in enumerate(self.chars)}

        data = torch.tensor(self.encode(text), dtype=torch.long)
        n = int(len(data) * (1 - val_frac))
        self.train = data[:n]
        self.val = data[n:]

    def encode(self, s):
        return [self.stoi[c] for c in s]

    def decode(self, ids):
        return "".join(self.itos[i] for i in ids)

    def get_batch(self, split, batch_size, block_size, device, generator=None):
        d = self.train if split == "train" else self.val
        ix = torch.randint(len(d) - block_size - 1, (batch_size,), generator=generator)
        x = torch.stack([d[i:i + block_size] for i in ix])
        y = torch.stack([d[i + 1:i + block_size + 1] for i in ix])
        return x.to(device), y.to(device)

    def chunks(self, split, block_size):
        """Cut a split into non-overlapping (x, y) chunks. Used for the final
        evaluation so every model is scored on exactly the same characters."""
        d = self.train if split == "train" else self.val
        n = (len(d) - 1) // block_size
        x = d[:n * block_size].view(n, block_size)
        y = d[1:n * block_size + 1].view(n, block_size)
        return x, y
