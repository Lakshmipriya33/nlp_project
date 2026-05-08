from collections import Counter
from typing import Dict, List

import torch
from torch.utils.data import Dataset


class BERTDisasterDataset(Dataset):
    def __init__(self, encodings: Dict[str, torch.Tensor], meta_features: torch.Tensor, labels: List[int]):
        self.encodings = encodings
        self.meta_features = meta_features
        self.labels = labels

    def __getitem__(self, idx: int):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["extra_features"] = self.meta_features[idx]
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self) -> int:
        return len(self.labels)


class LSTMDisasterDataset(Dataset):
    def __init__(self, sequences: torch.Tensor, meta_features: torch.Tensor, labels: List[int]):
        self.sequences = sequences
        self.meta_features = meta_features
        self.labels = labels

    def __getitem__(self, idx: int):
        return {
            "text": self.sequences[idx],
            "extra_features": self.meta_features[idx],
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }

    def __len__(self) -> int:
        return len(self.labels)


def build_vocab(texts: List[str], max_words: int = 10000) -> Dict[str, int]:
    words = " ".join(texts).split()
    word_counts = Counter(words)
    return {word: i + 1 for i, (word, _) in enumerate(word_counts.most_common(max_words))}


def text_to_sequence(text: str, vocab: Dict[str, int], max_len: int = 50) -> torch.Tensor:
    tokens = [vocab.get(word, 0) for word in text.split()]
    if len(tokens) < max_len:
        tokens += [0] * (max_len - len(tokens))
    else:
        tokens = tokens[:max_len]
    return torch.tensor(tokens, dtype=torch.long)
