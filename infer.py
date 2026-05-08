import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from transformers import BertTokenizer

from src.config import PipelineConfig
from src.data_utils import add_meta_features, get_stop_words, preprocess_text
from src.datasets import BERTDisasterDataset, LSTMDisasterDataset, text_to_sequence
from src.models import DeepDisasterModel, DisasterBERT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference for LSTM and BERT models")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Directory containing input CSV")
    parser.add_argument("--input-file", type=str, default="test.csv", help="CSV file for inference")
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"), help="Directory with trained artifacts")
    parser.add_argument("--outputs-dir", type=Path, default=Path("outputs"), help="Directory for predictions")
    return parser.parse_args()


def infer_lstm(
    df: pd.DataFrame,
    cfg: PipelineConfig,
    artifacts_dir: Path,
    device: torch.device,
) -> List[int]:
    with (artifacts_dir / "lstm_vocab.json").open("r", encoding="utf-8") as f:
        vocab: Dict[str, int] = json.load(f)

    sequences = torch.stack([text_to_sequence(t, vocab, cfg.lstm_max_len) for t in df["processed_text"].tolist()])
    meta = torch.tensor(df[cfg.meta_columns].values, dtype=torch.float)
    labels = df["target"].tolist() if "target" in df.columns else [0] * len(df)
    dataset = LSTMDisasterDataset(sequences, meta, labels)
    loader = DataLoader(dataset, batch_size=cfg.lstm_batch_size, shuffle=False)

    model = DeepDisasterModel(vocab_size=len(vocab) + 1, num_extra_features=len(cfg.meta_columns)).to(device)
    model.load_state_dict(torch.load(artifacts_dir / "best_disaster_bi_lstm.pth", map_location=device))
    model.eval()

    preds: List[int] = []
    with torch.no_grad():
        for batch in loader:
            text_input = batch["text"].to(device)
            extra_features = batch["extra_features"].to(device)
            outputs = model(text_input, extra_features)
            pred = torch.argmax(outputs, dim=1).cpu().tolist()
            preds.extend(pred)
    return preds


def infer_bert(
    df: pd.DataFrame,
    cfg: PipelineConfig,
    artifacts_dir: Path,
    device: torch.device,
) -> List[int]:
    tokenizer = BertTokenizer.from_pretrained(cfg.bert_model_name)
    encodings = tokenizer(
        df["text"].tolist(),
        truncation=True,
        padding=True,
        max_length=cfg.max_seq_len,
        return_tensors="pt",
    )
    meta = torch.tensor(df[cfg.meta_columns].values, dtype=torch.float)
    labels = df["target"].tolist() if "target" in df.columns else [0] * len(df)
    dataset = BERTDisasterDataset(encodings, meta, labels)
    loader = DataLoader(dataset, batch_size=cfg.bert_batch_size, shuffle=False)

    model = DisasterBERT(num_extra_features=len(cfg.meta_columns), bert_model_name=cfg.bert_model_name).to(device)
    model.load_state_dict(torch.load(artifacts_dir / "best_disaster_bert.pth", map_location=device))
    model.eval()

    preds: List[int] = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            extra_features = batch["extra_features"].to(device)
            outputs = model(input_ids, attention_mask, extra_features)
            pred = torch.argmax(outputs, dim=1).cpu().tolist()
            preds.extend(pred)
    return preds


def summarize_results(df: pd.DataFrame, lstm_preds: List[int], bert_preds: List[int]) -> Tuple[pd.DataFrame, Dict[str, float]]:
    result_df = pd.DataFrame()
    if "id" in df.columns:
        result_df["id"] = df["id"]
    result_df["lstm_pred"] = lstm_preds
    result_df["bert_pred"] = bert_preds
    result_df["ensemble_pred"] = [b if l != b else l for l, b in zip(lstm_preds, bert_preds)]

    summary: Dict[str, float] = {}
    if "target" in df.columns:
        y_true = df["target"].tolist()
        summary["lstm_accuracy"] = accuracy_score(y_true, lstm_preds)
        summary["bert_accuracy"] = accuracy_score(y_true, bert_preds)
        summary["ensemble_accuracy"] = accuracy_score(y_true, result_df["ensemble_pred"].tolist())

    return result_df, summary


def main() -> None:
    args = parse_args()
    cfg = PipelineConfig()
    args.outputs_dir.mkdir(parents=True, exist_ok=True)

    input_path = args.data_dir / args.input_file
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    with (args.artifacts_dir / "meta_scaler.pkl").open("rb") as f:
        scaler = pickle.load(f)

    stop_words = get_stop_words()
    data_df = pd.read_csv(input_path)
    data_df = add_meta_features(data_df, stop_words)
    data_df["processed_text"] = data_df["text"].apply(lambda t: preprocess_text(t, stop_words))
    data_df.loc[:, cfg.meta_columns] = scaler.transform(data_df[cfg.meta_columns])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lstm_preds = infer_lstm(data_df, cfg, args.artifacts_dir, device)
    bert_preds = infer_bert(data_df, cfg, args.artifacts_dir, device)

    result_df, summary = summarize_results(data_df, lstm_preds, bert_preds)

    output_csv = args.outputs_dir / "inference_results.csv"
    output_json = args.outputs_dir / "inference_summary.json"

    result_df.to_csv(output_csv, index=False)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Inference complete.")
    print("Saved predictions:", output_csv)
    print("Saved summary:", output_json)
    print("Preview:")
    print(result_df.head(10).to_string(index=False))
    if summary:
        print("Metrics:")
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
