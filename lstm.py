from typing import Tuple
import sys
import torch
import torch.nn.functional as F
import torch.nn as nn

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin

class MyLSTM(ModelMixin, ConfigMixin):

    """Implementation of the standard LSTM.

    Parameters
    ----------
    input_size : int
        Number of input features
    hidden_size : int
        Number of hidden/memory cells.
    pred_len: int
        Look back window length
    num_layers: int 
        Number of LSTM layers used
    batch_first : bool, optional
        If True, expects the batch inputs to be of shape [batch, seq, features] otherwise, the
        shape has to be [seq, batch, features], by default True.
    initial_forget_bias : int, optional
        Value of the initial forget gate bias, by default 0
    """
    @register_to_config
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 pred_len: int,
                 num_layers: int = 2,
                 batch_first: bool = True,
                 output_dropout_prob: float = 0.2
                 ):
        super(MyLSTM, self).__init__()
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=batch_first,
                                  )
        
        self.pred_len = pred_len
        self.dropout = torch.nn.Dropout(p=output_dropout_prob)
        self.out_proj = nn.Linear(hidden_size, input_size)

    def forward(self, x_d: torch.Tensor) -> torch.Tensor:
        preds = []

        # warm-up with given sequence
        output, (h, c) = self.lstm(x_d)

        # start autoregression from last input
        last_input = x_d[:, -1:, :]

        for _ in range(self.pred_len):
            output, (h, c) = self.lstm(last_input, (h, c))
            step_pred = self.out_proj(self.dropout(output[:,-1,:]))  # project hidden state
            preds.append(step_pred.unsqueeze(1))
            last_input = step_pred.unsqueeze(1)  # feed prediction as input

        preds = torch.cat(preds, dim=1)  # [batch, pred_steps, output_dim]

        return preds