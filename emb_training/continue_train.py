import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset
import deepspeed
from model import Word2VecNegSampling

class WordDataset(Dataset):
    def __init__(self, np_path, vocab_size, num_neg=5):
        self.data = np.load(np_path)
        self.vocab_size = vocab_size
        self.num_neg = num_neg

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        center, pos = self.data[idx]
        negs = np.random.randint(0, self.vocab_size, size=(self.num_neg,))
        return torch.tensor(center), torch.tensor(pos), torch.tensor(negs)

def train():
    with open("preprocessed/meta.txt") as f:
        vocab_size = int(f.readline())

    model = Word2VecNegSampling(vocab_size, 600)

    # Load existing weights
    model.load_state_dict(torch.load("embedding_model.pth"))

    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config_params="ds_config.json"
    )

    dataset = WordDataset("preprocessed/train_pairs.npy", vocab_size)
    dataloader = DataLoader(dataset, batch_size=2048, shuffle=True, num_workers=4)

    for epoch in range(10):  # Continue for 10 more epochs
        for i, (center, pos, negs) in enumerate(dataloader):
            center = center.to(model_engine.device)
            pos = pos.to(model_engine.device)
            negs = negs.to(model_engine.device)

            loss = model_engine(center, pos, negs)
            model_engine.backward(loss)
            model_engine.step()

            if i % 100 == 0:
                print(f"[CONTINUE] Epoch {epoch} | Step {i} | Loss: {loss.item():.4f}")

if __name__ == "__main__":
    train()

continue_train_path = "/mnt/data/continue_training.py"
with open(continue_train_path, "w") as f:
    f.write(continue_train_code)
continue_train_path
