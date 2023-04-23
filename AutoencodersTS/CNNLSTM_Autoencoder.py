'''
Documentation for TimeSeriesCNNLSTMAutoencoder class:


This class is a Pytorch implementation of a time series autoencoder 
using a combination of CNN and LSTM layers. The autoencoder takes as 
input a time series of length "input_size" and outputs a time series 
of length "output_size".


The architecture consists of two main components: an encoder and a decoder. 
The encoder first passes the input through a CNN layer, then passes 
it through an LSTM layer to obtain a latent vector. The decoder then takes 
this latent vector and passes it through an LSTM layer and a reversed CNN 
layer to reconstruct the input.


The class takes in the following parameters:



input_size: the length of the input time series

hidden_size: the number of features in the LSTM layers

output_size: the length of the output time series

num_layers: the number of LSTM layers to use in 
            both the encoder and decoder


The forward() method takes in a tensor of shape 
(batch_size, input_size, 1) and returns a tensor of shape (batch_size, output_size).


The architecture and hyperparameters used in this implementation are 
based on the paper "Time Series Anomaly Detection 
with Variational Autoencoders" by Malhotra et al. (2016).
'''
import torch
import torch.nn as nn


class TimeSeriesCNNLSTMAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super().__init__()

        self.cnn_encoder = nn.Sequential(
            nn.Conv1d(input_size, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )
        self.lstm_encoder = nn.LSTM(
            64, hidden_size, num_layers=num_layers, batch_first=True)

        self.lstm_decoder = nn.LSTM(
            hidden_size, hidden_size, num_layers=num_layers, batch_first=True)

        self.cnn_decoder = nn.Sequential(
            nn.ConvTranspose1d(hidden_size, 64, kernel_size=3, stride=2,
                               padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(64, 32, kernel_size=3, stride=2,
                               padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(32, output_size, kernel_size=3, padding=1),
        )

        self.pool = nn.AdaptiveMaxPool1d(output_size)

        self.fc = nn.Linear(output_size, output_size)

    def forward(self, x):
        # Pass input through encoder and get latent space representation.
        cnn_encoded = self.cnn_encoder(x)
        cnn_encoded = cnn_encoded.permute(0, 2, 1)  # swap dimensions for LSTM
        _, (lstm_encoded, _) = self.lstm_encoder(cnn_encoded)
        # use last hidden state as latent vector
        lstm_encoded = lstm_encoded[-1]

        # Repeat latent vector to match sequence length.
        lstm_decoded = lstm_encoded.unsqueeze(1).repeat(1, output_size, 1)

        # Pass decoded sequence through decoder.
        lstm_decoded, _ = self.lstm_decoder(lstm_decoded)
        cnn_decoded = lstm_decoded.permute(0, 2, 1)  # swap dimensions for CNN
        cnn_decoded = self.cnn_decoder(cnn_decoded)
        pooled = self.pool(cnn_decoded).squeeze(2)
        fc_decoded = self.fc(pooled)

        return fc_decoded
