import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from transformers import BertTokenizer

from src.config import PipelineConfig
from src.data_utils import add_meta_features, get_stop_words, preprocess_text
from src.datasets import (
    BERTDisasterDataset,
    LSTMDisasterDataset,
    build_vocab,
    text_to_sequence,
)
from src.models import BidirectionalLSTMModel, DeepDisasterModel, DisasterBERT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LSTM and BERT disaster classifiers")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Directory containing train.csv")
    parser.add_argument("--train-file", type=str, default="train.csv", help="Training CSV file name")
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"), help="Directory to store models/artifacts")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_data(train_path: Path, cfg: PipelineConfig) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler, Dict[str, int]]:
    stop_words = get_stop_words()
    train_df = pd.read_csv(train_path)
    train_df = add_meta_features(train_df, stop_words)

    train_split_df, val_df = train_test_split(
        train_df,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=train_df["target"],
    )

    train_split_df = train_split_df.copy()
    val_df = val_df.copy()

    train_split_df["processed_text"] = train_split_df["text"].apply(lambda t: preprocess_text(t, stop_words))
    val_df["processed_text"] = val_df["text"].apply(lambda t: preprocess_text(t, stop_words))

    scaler = StandardScaler()
    train_split_df.loc[:, cfg.meta_columns] = scaler.fit_transform(train_split_df[cfg.meta_columns])
    val_df.loc[:, cfg.meta_columns] = scaler.transform(val_df[cfg.meta_columns])

    vocab = build_vocab(train_split_df["processed_text"].tolist(), cfg.lstm_vocab_size)
    return train_split_df, val_df, scaler, vocab


def train_lstm(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    cfg: PipelineConfig,
    device: torch.device,
    vocab: Dict[str, int],
    artifacts_dir: Path,
) -> Dict[str, float]:
    train_sequences = torch.stack(
        [text_to_sequence(t, vocab, cfg.lstm_max_len) for t in train_df["processed_text"].tolist()]
    )
    val_sequences = torch.stack([text_to_sequence(t, vocab, cfg.lstm_max_len) for t in val_df["processed_text"].tolist()])

    train_meta = torch.tensor(train_df[cfg.meta_columns].values, dtype=torch.float)
    val_meta = torch.tensor(val_df[cfg.meta_columns].values, dtype=torch.float)

    train_dataset = LSTMDisasterDataset(train_sequences, train_meta, train_df["target"].tolist())
    val_dataset = LSTMDisasterDataset(val_sequences, val_meta, val_df["target"].tolist())

    train_loader = DataLoader(train_dataset, batch_size=cfg.lstm_batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.lstm_batch_size, shuffle=False)

    model = DeepDisasterModel(vocab_size=len(vocab) + 1, num_extra_features=len(cfg.meta_columns)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lstm_lr)
    criterion = nn.CrossEntropyLoss()

    best_accuracy = 0.0
    best_path = artifacts_dir / "best_disaster_lstm.pt"

    for epoch in range(cfg.lstm_epochs):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            text_input = batch["text"].to(device)
            extra_features = batch["extra_features"].to(device)
            labels = batch["label"].to(device)
            outputs = model(text_input, extra_features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        _, val_accuracy = evaluate_lstm(model, val_loader, criterion, device)
        print(f"[LSTM] Epoch {epoch + 1}/{cfg.lstm_epochs} | Val Accuracy: {val_accuracy:.4f}")
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), best_path)

    return {"best_val_accuracy": best_accuracy, "checkpoint": str(best_path)}

def train_bi_lstm(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    cfg: PipelineConfig,
    device: torch.device,
    vocab: Dict[str, int],
    artifacts_dir: Path,
) -> Dict[str, float]:
    train_sequences = torch.stack(
        [text_to_sequence(t, vocab, cfg.lstm_max_len) for t in train_df["processed_text"].tolist()]
    )
    val_sequences = torch.stack([text_to_sequence(t, vocab, cfg.lstm_max_len) for t in val_df["processed_text"].tolist()])

    train_meta = torch.tensor(train_df[cfg.meta_columns].values, dtype=torch.float)
    val_meta = torch.tensor(val_df[cfg.meta_columns].values, dtype=torch.float)

    train_dataset = LSTMDisasterDataset(train_sequences, train_meta, train_df["target"].tolist())
    val_dataset = LSTMDisasterDataset(val_sequences, val_meta, val_df["target"].tolist())

    train_loader = DataLoader(train_dataset, batch_size=cfg.lstm_batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.lstm_batch_size, shuffle=False)

    model = BidirectionalLSTMModel(vocab_size=len(vocab) + 1, num_extra_features=len(cfg.meta_columns)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lstm_lr)
    criterion = nn.CrossEntropyLoss()

    best_accuracy = 0.0
    best_path = artifacts_dir / "best_disaster_bi_lstm.pt"

    for epoch in range(cfg.lstm_epochs):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            text_input = batch["text"].to(device)
            extra_features = batch["extra_features"].to(device)
            labels = batch["label"].to(device)
            outputs = model(text_input, extra_features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        _, val_accuracy = evaluate_lstm(model, val_loader, criterion, device)
        print(f"[BiLSTM] Epoch {epoch + 1}/{cfg.lstm_epochs} | Val Accuracy: {val_accuracy:.4f}")
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), best_path)

    return {"best_val_accuracy": best_accuracy, "checkpoint": str(best_path)}   


def evaluate_lstm(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for batch in loader:
            text_input = batch["text"].to(device)
            extra_features = batch["extra_features"].to(device)
            labels = batch["label"].to(device)
            outputs = model(text_input, extra_features)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            total_correct += torch.sum(preds == labels).item()
            total_samples += labels.size(0)
    avg_loss = total_loss / max(len(loader), 1)
    accuracy = total_correct / max(total_samples, 1)
    return avg_loss, accuracy


def train_bert(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    cfg: PipelineConfig,
    device: torch.device,
    artifacts_dir: Path,
) -> Dict[str, float]:
    tokenizer = BertTokenizer.from_pretrained(cfg.bert_model_name)
    train_encodings = tokenizer(
        train_df["text"].tolist(),
        truncation=True,
        padding=True,
        max_length=cfg.max_seq_len,
        return_tensors="pt",
    )
    val_encodings = tokenizer(
        val_df["text"].tolist(),
        truncation=True,
        padding=True,
        max_length=cfg.max_seq_len,
        return_tensors="pt",
    )

    train_meta = torch.tensor(train_df[cfg.meta_columns].values, dtype=torch.float)
    val_meta = torch.tensor(val_df[cfg.meta_columns].values, dtype=torch.float)

    train_dataset = BERTDisasterDataset(train_encodings, train_meta, train_df["target"].tolist())
    val_dataset = BERTDisasterDataset(val_encodings, val_meta, val_df["target"].tolist())

    train_loader = DataLoader(train_dataset, batch_size=cfg.bert_batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.bert_batch_size, shuffle=False)

    model = DisasterBERT(num_extra_features=len(cfg.meta_columns), bert_model_name=cfg.bert_model_name).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.bert_lr)
    criterion = nn.CrossEntropyLoss()

    best_accuracy = 0.0
    best_path = artifacts_dir / "best_disaster_bert.pt"

    for epoch in range(cfg.bert_epochs):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            meta = batch["extra_features"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids, attention_mask, meta)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        _, val_accuracy = evaluate_bert(model, val_loader, criterion, device)
        print(f"[BERT] Epoch {epoch + 1}/{cfg.bert_epochs} | Val Accuracy: {val_accuracy:.4f}")
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), best_path)

    return {"best_val_accuracy": best_accuracy, "checkpoint": str(best_path)}


def evaluate_bert(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            meta = batch["extra_features"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids, attention_mask, meta)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            total_correct += torch.sum(preds == labels).item()
            total_samples += labels.size(0)
    avg_loss = total_loss / max(len(loader), 1)
    accuracy = total_correct / max(total_samples, 1)
    return avg_loss, accuracy


def main() -> None:
    args = parse_args()
    cfg = PipelineConfig()
    set_seed(cfg.random_state)

    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.data_dir / args.train_file
    if not train_path.exists():
        raise FileNotFoundError(f"Training file not found: {train_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_df, val_df, scaler, vocab = prepare_data(train_path, cfg)

    lstm_metrics = train_lstm(train_df, val_df, cfg, device, vocab, args.artifacts_dir)
    bert_metrics = train_bert(train_df, val_df, cfg, device, args.artifacts_dir)

    with (args.artifacts_dir / "meta_scaler.pkl").open("wb") as f:
        pickle.dump(scaler, f)

    with (args.artifacts_dir / "lstm_vocab.json").open("w", encoding="utf-8") as f:
        json.dump(vocab, f)

    run_summary = {
        "device": str(device),
        "lstm": lstm_metrics,
        "bert": bert_metrics,
        "meta_columns": cfg.meta_columns,
        "bert_model_name": cfg.bert_model_name,
        "lstm_max_len": cfg.lstm_max_len,
    }

    with (args.artifacts_dir / "training_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(run_summary, f, indent=2)

    print("Training complete. Artifacts saved to:", args.artifacts_dir)
    print(json.dumps(run_summary, indent=2))


if __name__ == "__main__":
    main()
