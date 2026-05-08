from dataclasses import dataclass, field
from typing import List


@dataclass
class PipelineConfig:
    meta_columns: List[str] = field(
        default_factory=lambda: [
            "word_count",
            "unique_word_count",
            "stop_word_count",
            "url_count",
            "mean_word_length",
        ]
    )
    random_state: int = 42
    test_size: float = 0.2
    bert_model_name: str = "bert-base-uncased"
    max_seq_len: int = 128
    lstm_max_len: int = 50
    lstm_vocab_size: int = 10000
    lstm_batch_size: int = 32
    bert_batch_size: int = 16
    lstm_epochs: int = 5
    bert_epochs: int = 3
    lstm_lr: float = 1e-3
    bert_lr: float = 2e-5
