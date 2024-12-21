import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TemporalDataset(Dataset):
    def __init__(self, X, y=None):
        """
        X: (N, T, D) where N is number of sequences, T is sequence length, D is dimensionality
        y: (N,) or None
        """
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        else:
            return self.X[idx]

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, latent_dim, num_layers, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc_mu = nn.Linear(hidden_size, latent_dim)
        self.fc_logvar = nn.Linear(hidden_size, latent_dim)
        self.embedding = nn.Embedding(num_classes, latent_dim)

    def forward(self, x, labels):
        _, (h_n, _) = self.lstm(x)
        h_n = h_n[-1]  # Get last hidden state
        mu = self.fc_mu(h_n)
        logvar = self.fc_logvar(h_n)
        z = self.reparameterize(mu, logvar)
        emb = self.embedding(labels)
        return z, mu, logvar, emb

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_size, output_size, num_layers, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(latent_dim + num_classes, hidden_size, num_layers, batch_first=True)
        self.fc_out = nn.Linear(hidden_size, output_size)

    def forward(self, z, emb, seq_len):
        z_emb = torch.cat([z.unsqueeze(1).repeat(1, seq_len, 1), emb.unsqueeze(1).repeat(1, seq_len, 1)], dim=-1)
        outputs, _ = self.lstm(z_emb)
        outputs = self.fc_out(outputs)
        return outputs

class CVAE(nn.Module):
    def __init__(self, input_size, hidden_size, latent_dim, num_layers, num_classes):
        super().__init__()
        self.encoder = Encoder(input_size, hidden_size, latent_dim, num_layers, num_classes)
        self.decoder = Decoder(latent_dim, hidden_size, input_size, num_layers, num_classes)

    def forward(self, x, labels, seq_len):
        z, mu, logvar, emb = self.encoder(x, labels)
        recon_x = self.decoder(z, emb, seq_len)
        return recon_x, mu, logvar, emb

def loss_function(recon_x, x, mu, logvar):
    MSE = F.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return MSE + KLD

def train(model, optimizer, epoch, dataloader):
    model.train()
    train_loss = 0
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data, target, data.size(1))
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % 50 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(dataloader.dataset)} ({100. * batch_idx / len(dataloader):.0f}%)]\tLoss: {loss.item() / len(data):.6f}')
    print(f'====> Epoch: {epoch} Average loss: {train_loss / len(dataloader.dataset):.4f}')

def test(model, epoch, dataloader):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            recon_batch, mu, logvar = model(data, target, data.size(1))
            test_loss += loss_function(recon_batch, data, mu, logvar).item()
    test_loss /= len(dataloader.dataset)
    print(f'====> Test set loss: {test_loss:.4f}')

if __name__ == "__main__":
    # Hyperparameters
    batch_size = 64
    input_size = 2  # Two variables per time step
    hidden_size = 256
    latent_dim = 20
    num_epochs = 10
    learning_rate = 1e-3
    num_layers = 2
    num_classes = 10  # Number of classes for your task
    seq_len = 1000  # Sequence length

    # Dummy data generation
    X = torch.randn(1000, seq_len, input_size)
    y = torch.randint(0, num_classes, (1000,))

    # Create dataset and dataloaders
    train_dataset = TemporalDataset(X[:800], y[:800])
    test_dataset = TemporalDataset(X[800:], y[800:])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model, optimizer, and scheduler
    model = CVAE(input_size, hidden_size, latent_dim, num_layers, num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(1, num_epochs + 1):
        train(model, optimizer, epoch, train_loader)
        test(model, epoch, test_loader)
