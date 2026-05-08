import torch
import torch.nn as nn
from transformers import BertModel


class DeepDisasterModel(nn.Module):
    def __init__(self, vocab_size: int, num_extra_features: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 100)
        self.lstm = nn.LSTM(100, 64, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(64 + num_extra_features, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )

    def forward(self, text_input: torch.Tensor, extra_features: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(text_input)
        _, (hidden, _) = self.lstm(embedded)
        lstm_out = hidden[-1]
        combined = torch.cat((lstm_out, extra_features), dim=1)
        return self.classifier(combined)
    
class BidirectionalLSTMModel(nn.Module):
    def __init__(self, vocab_size: int, num_extra_features: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 100)
        
        # 1. Set bidirectional=True
        self.lstm = nn.LSTM(100, 64, batch_first=True, bidirectional=True)
        
        # 2. Update the input dimension of the first Linear layer
        # It is now (64 * 2) from LSTM + num_extra_features
        self.classifier = nn.Sequential(
            nn.Linear(128 + num_extra_features, 32), 
            nn.ReLU(),
            nn.Linear(32, 2),
        )

    def forward(self, text_input: torch.Tensor, extra_features: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(text_input)
        
        # _, (hidden, _) = self.lstm(embedded)
        # For Bi-LSTM, 'hidden' contains the last state for both directions
        output, (hidden, _) = self.lstm(embedded)
        
        # 3. Concatenate the final forward and backward hidden states
        # hidden[-2] is the last state of the forward pass
        # hidden[-1] is the last state of the backward pass
        lstm_out = torch.cat((hidden[-2], hidden[-1]), dim=1)
        
        combined = torch.cat((lstm_out, extra_features), dim=1)
        return self.classifier(combined)


class DisasterBERT(nn.Module):
    def __init__(self, num_extra_features: int, bert_model_name: str = "bert-base-uncased"):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size + num_extra_features, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        extra_features: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = self.dropout(outputs.pooler_output)
        combined = torch.cat((pooled_output, extra_features), dim=1)
        return self.classifier(combined)
