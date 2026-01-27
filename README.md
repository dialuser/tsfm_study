# Time Series Foundation Model Study

This repository contains source codes used for *Zero-shot Forecasting of Streamflow Using Time Series Foundation Models: Are We There Yet?*

### 1d-ahead forecast

- `lstm_camelsdaily.py`: main code for training 1d-ahead LSTM model
- `camels_dataloader_daily.py`: daily data loader file
- `lstm.py`: LSTM model
- `config_lstm_camelsdaily.yaml`: YAML configuration file for 1d-ahead training

### 3H-ahead forecast
- `lstm_main.py`: main code for training 3H-ahead LSTM model
- `camels_dataloader.py`: 3H data loader file
- `lstm.py`: LSTM model
- `config_lstm_camels.yaml`: YAML configuratoin file for 3H-ahead training


Dependencies:

- pytorch, 2.3.1
- python, 3.10.0
- diffusers, 0.29.2
- accelerate, 0.31.0
- omegaconf,  2.3.0

