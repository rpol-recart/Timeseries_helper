import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()

        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        seq_len = x.size(1)
        _, (hidden_state, _) = self.lstm(x)
        x = self.fc(hidden_state)

        return seq_len, x.permute(1, 0, 2)


class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Decoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.fc = nn.Linear(latent_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.fc2 = nn.Linear(hidden_dim, input_dim)

    def forward(self, seq_len, latent):
        hidden = (torch.randn(1, 1, self.hidden_dim).to('cuda'),
                  torch.randn(1, 1, self.hidden_dim).to('cuda'))
        x = self.fc(latent)

        x = x.repeat(1, seq_len, 1)
        x, hidden = self.lstm(x, hidden)
        x = self.fc2(x)

        return x


class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Autoencoder, self).__init__()

        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(input_dim, hidden_dim, latent_dim)

    def forward(self, x):
        # Encoder
        seq_len, hidden_state = self.encoder(x)

        # Decoder
        output = self.decoder(seq_len, hidden_state)

        return output
