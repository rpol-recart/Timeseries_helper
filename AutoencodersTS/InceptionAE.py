import torch.nn as nn
from InceptionTime import InceptionBlock, InceptionTransposeBlock


class Latent(nn.Module):
    """This class defines a latent module that inherits from nn.Module in Pytorch. It has three attributes:\n\n    Attributes:
        input_channels (int): Number of channels in input tensor. Default is 1.
        output_channels (int): Number of channels in output of Conv1d layer. Default is 178.
        output_size (int): Specific output size of the AdaptiveAvgPool1d layer. Default is 1.

    Methods:
        __init__(self, input_channels=1, output_channels=178, output_size=1):
            This method initializes the latent module with the specified attributes.

        forward(self, X, indices):
            This method takes in a tensor X and indices, and passes X through the initialized layers to produce 
            a latent representation of the input tensor. The latent tensor is then "unpooled" back to the original
            output size using the deconvolutional layer, and finally reshaped to produce the output.
    """

    def __init__(self, input_channels=1, output_channels=178, output_size=1):
        super(Latent, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(output_size=output_size)
        self.flatten = nn.Flatten(1, 2)
        self.deconv = nn.Conv1d(
            in_channels=input_channels, out_channels=output_channels, kernel_size=1)

    def forward(self, X, indices):
        x = self.avg_pool(X)
        latent = self.flatten(x)
        unpull = self.deconv(latent.unsqueeze(-1)).squeeze(-1)
        unpull = unpull.view(-1, unpull.size(2), unpull.size(1))
        return latent, unpull, indices


class Autoencoder(nn.Module):
    """
        A PyTorch module for an autoencoder neural network architecture.
        The autoencoder consists of an InceptionBlock-based encoder,
        a Latent module for bottlenecking the encoded data, and an
        InceptionTransposeBlock-based decoder.

        Parameters:
        ----------
        input_size : int
            The size of the input to the network, in pixels.
        hidden_size : int
            The number of channels to use for the encoder's bottleneck.
        kernel_size : int
            The size of the convolutional kernels to use in the network.

        Methods:
        -------
        forward(x)
            Runs the forward pass of the autoencoder on the input tensor x.
            Returns the reconstructed output tensor.

    """

    def __init__(self, input_size, hidden_size, kernel_size):
        super().__init__()
        self.encoder = InceptionBlock(
            in_channels=1,
            n_filters=8,
            kernel_sizes=[9, 19, 39],
            bottleneck_channels=32,
            use_residual=True,
            activation=nn.ReLU(),
            return_indices=True
        )

        self.decoder = InceptionTransposeBlock(
            in_channels=32,
            out_channels=1,
            kernel_sizes=[9, 19, 39],
            bottleneck_channels=32,
            use_residual=True,
            activation=nn.ReLU()
        )
        self.latent = Latent()

    def forward(self, x):
        x = x.unsqueeze(1)
        encoded = self.encoder(x)
        _, x, indices = self.latent(encoded[0], encoded[1])
        decoded = self.decoder(x, indices)
        return decoded.squeeze(1)
