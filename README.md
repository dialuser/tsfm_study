# Time Series Foundation Model Study

This repository contains source codes used for Sun, A.Y. and Sun, A.A, *Zero-shot Forecasting of Streamflow Using Time Series Foundation Models: Are We There Yet?*

### Univeriate  LSTM Experiments

1. 1d-ahead forecast

- `lstm_camelsdaily.py`: main code for training 1d-ahead LSTM model
- `camels_dataloader_daily.py`: daily data loader file
- `lstm.py`: LSTM model
- `config_lstm_camelsdaily.yaml`: YAML configuration file for 1d-ahead training

2. 3H-ahead forecast
- `lstm_main.py`: main code for training 3H-ahead LSTM model
- `camels_dataloader.py`: 3H data loader file
- `lstm.py`: LSTM model
- `config_lstm_camels.yaml`: YAML configuratoin file for 3H-ahead training

### Univariate Experiments

1. 1d-ahead forecast

- `tsfm/1d/sundial_camels_daily.py`: main code for 1D-ahead univariate zero-shot benchmarking using Sundial, Chronos, TTM, and MOIRAI
- `tsfm/1d/camels_dataloader_daily.py`: data loader routines
- `tsfm/1d/config_sundial_camelsdaily.yaml`: configuration file 


2. 3H-ahead forecast

- `tsfm/3h/sundial_camels.py`: main code for 3H-ahead univariate zero-shot benchmarking using Sundial, Chronos, TTM, and MOIRAI
- `tsfm/3h/camels_dataloader.py`: data loader routines
- `tsfm/3h/config_sundail_camels.yaml`: configuration file


Dependencies:

- pytorch, 2.3.1
- python, 3.10.0
- diffusers, 0.29.2
- accelerate, 0.31.0
- omegaconf,  2.3.0

