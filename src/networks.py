import torch
from torch import nn

torch.manual_seed(0)  # Set for testing purposes, please do not change!


class SimpleNet(nn.Module):
    '''
    Simply do a linear network
    '''

    def __init__(self, im_chan, n_classes=1):
        super(SimpleNet, self).__init__()
        self.simple = nn.Sequential(
            nn.Linear(im_chan, n_classes)
        )

    def forward(self, image):
        '''
        :param image: batch_size, 1, 28, 28 images
        :return:
        '''
        return self.simple.forward(image.flatten(start_dim=1))

    def get_embedding(self, x):
        return self.forward(x)


class EmbeddingNet(nn.Module):
    '''
    EmbeddingNet Class
    Values:
        im_chan: the number of channels in the images, fitted for the dataset used, a scalar
              (MNIST is black-and-white, so 1 channel is your default)
    hidden_dim: the inner dimension, a scalar
    '''

    def __init__(self, im_chan=1, hidden_dim=16, projection_dim=2):
        super(EmbeddingNet, self).__init__()
        self.disc = nn.Sequential(
            self.make_disc_block(im_chan, hidden_dim),
            self.make_disc_block(hidden_dim, hidden_dim * 2),
            self.make_disc_block(hidden_dim * 2, projection_dim, final_layer=True),
        )

    @staticmethod
    def _get_flattened_shape(kernel_size, input_channels):
        return ((kernel_size + 1) ** 2) * input_channels

    def make_disc_block(self, input_channels, output_channels, kernel_size=4, stride=2, final_layer=False):
        '''
        Function to return a sequence of operations corresponding to a EmbeddingNet block of DCGAN,
        corresponding to a convolution, a batchnorm (except for in the last layer), and an activation.
        Parameters:
            input_channels: how many channels the input feature representation has
            output_channels: how many channels the output feature representation should have
            kernel_size: the size of each convolutional filter, equivalent to (kernel_size, kernel_size)
            stride: the stride of the convolution
            final_layer: a boolean, true if it is the final layer and false otherwise
                      (affects activation and batchnorm)
        '''
        #     Steps:
        #       1) Add a convolutional layer using the given parameters.
        #       2) Do a batchnorm, except for the last layer.
        #       3) Follow each batchnorm with a LeakyReLU activation with slope 0.2.

        # Build the neural block
        if not final_layer:
            return nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size, stride),
                nn.BatchNorm2d(output_channels),
                nn.LeakyReLU(negative_slope=0.2),
            )
        else:  # Final Layer
            return nn.Sequential(
                nn.Flatten(),
                nn.Linear(self._get_flattened_shape(kernel_size, input_channels), input_channels),
                nn.LeakyReLU(negative_slope=0.2),
                nn.Linear(input_channels, input_channels),
                nn.LeakyReLU(negative_slope=0.2),
                nn.Linear(input_channels, output_channels)
            )

    def forward(self, image):
        '''
        Function for completing a forward pass of the EmbeddingNet: Given an image tensor,
        returns a 1-dimension tensor representing the digit
        Parameters:
            image: a flattened image tensor with dimension (im_dim)
        '''
        disc_pred = self.disc(image)
        return disc_pred

    def get_embedding(self, x):
        return self.forward(x)


class ClassificationNet(nn.Module):
    """Simple linear classier.
    Returns logits"""

    def __init__(self, projection_dim=2, classes=10, hidden_dim=16, im_chan=1):
        super(ClassificationNet, self).__init__()
        self.embedding_net = EmbeddingNet(im_chan=im_chan, projection_dim=projection_dim, hidden_dim=hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(projection_dim, classes)
        )

    def forward(self, x):
        embedding = self.embedding_net(x)
        return self.classifier(embedding)

    def get_embedding(self, x):
        return self.embedding_net(x)


class Autoencoder(nn.Module):
    def __init__(self, projection_dim=2, im_chan=1, hidden_dim=16):
        super(Autoencoder, self).__init__()
        self.embedding_net = EmbeddingNet(im_chan=im_chan, projection_dim=projection_dim, hidden_dim=hidden_dim)
        # Build the neural network
        self.decoder = nn.Sequential(
            self.make_decode_block(projection_dim, hidden_dim * 4),
            self.make_decode_block(hidden_dim * 4, hidden_dim * 2, kernel_size=4, stride=1),
            self.make_decode_block(hidden_dim * 2, hidden_dim),
            self.make_decode_block(hidden_dim, im_chan, kernel_size=4, final_layer=True),
        )

    def make_decode_block(self, input_channels, output_channels, kernel_size=3, stride=2, final_layer=False):
        '''
        Function to return a sequence of operations corresponding to a generator block of DCGAN,
        corresponding to a transposed convolution, a batchnorm (except for in the last layer), and an activation.
        Parameters:
            input_channels: how many channels the input feature representation has
            output_channels: how many channels the output feature representation should have
            kernel_size: the size of each convolutional filter, equivalent to (kernel_size, kernel_size)
            stride: the stride of the convolution
            final_layer: a boolean, true if it is the final layer and false otherwise
                      (affects activation and batchnorm)
        '''

        #     Steps:
        #       1) Do a transposed convolution using the given parameters.
        #       2) Do a batchnorm, except for the last layer.
        #       3) Follow each batchnorm with a ReLU activation.
        #       4) If its the final layer, use a Tanh activation after the deconvolution.

        # Build the neural block
        if not final_layer:
            return nn.Sequential(
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size=kernel_size, stride=stride),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(inplace=True)
            )
        else:  # Final Layer
            return nn.Sequential(
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size=kernel_size, stride=stride),
                nn.Tanh()
            )

    def forward(self, image):
        '''
        Function for completing a foward pas of the Autoencoder: Given an image tensor, returns a reconstructed image tensor
        :param image:
        :return:
        '''
        encoded = self.embedding_net(image)
        reconstructed = self.decoder(encoded.unsqueeze(-1).unsqueeze(-1))
        return reconstructed

    def get_embedding(self, x):
        return self.embedding_net(x)


class SiameseNet(nn.Module):
    def __init__(self, embedding_net):
        super(SiameseNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        return output1, output2

    def get_embedding(self, x):
        return self.embedding_net(x)


class SimCLRNet(nn.Module):
    def __init__(self, embedding_net):
        super(SimCLRNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        # stack each set of augmented images and put through the same encoder, similar to siamese net
        representations = torch.cat((x1, x2), axis=0)
        output = self.embedding_net(representations)
        # breakpoint()
        return output

    def get_embedding(self, x):
        return self.embedding_net(x)
