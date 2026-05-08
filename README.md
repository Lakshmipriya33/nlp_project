# Disaster Tweets LSTM + BERT Pipeline
It trains:
- `DeepDisasterModel` (LSTM + engineered meta-features)
- 'BidirectionalLSTMModel` (Bi- directional LSTM + engineered meta-features)
- `DisasterBERT` (BERT + engineered meta-features)

It also runs an inference pipeline that outputs both model predictions and an ensemble column.

## Folder Layout

- `train.py`: trains LSTM and BERT models, saves best checkpoints and metrics
- `infer.py`: loads trained models and runs inference, writes output files
- `data_utils.py`: feature engineering and text preprocessing
- `datasets.py`: PyTorch dataset classes and vocabulary utilities
- `models.py`: model definitions 


## Training

### Prerequisites

- Install dependencies:

```bash
pip install -r requirements.txt
```

### Prepare Input And Output Folders

Before training, make sure the following structure exists:

```text
disaster_training_pipeline/
	data/
		train.csv
	artifacts/
```

- Create a `data` directory and place `train.csv` inside it.
- Create an `artifacts` directory to store training outputs.

### Windows Training Script

Save this as `run_train.bat` (or use the existing file):

```bat
@echo off
setlocal
python -m train --data-dir data --train-file train.csv --artifacts-dir artifacts
endlocal
```

Run training with:

```bat
run_train.bat
```

### Expected Outputs

After a successful run, the `artifacts` folder will contain trained model checkpoints and training metrics.
