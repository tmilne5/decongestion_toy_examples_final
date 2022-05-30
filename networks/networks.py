import torch.nn as nn
import torch


"""A simple fully connected discriminator. It has a scalar output, 2 layers, and dim hidden units"""

class Discriminator(nn.Module):

    def __init__(self, dim, relu=False, source_dim=2):
        super(Discriminator, self).__init__()

        self.nonlin = nn.ReLU() if relu else nn.Tanh()
        self.main = nn.Sequential(nn.Linear(source_dim, dim), self.nonlin, nn.Linear(dim, 1))

    def forward(self, inputs):
        output = self.main(inputs)
        return output

"""A simple affine generator, mapping standard normal noise z in DIM dimensional latent space to DIM"""

class Affine(nn.Module):
    def __init__(self, latent_dim):
        super(Affine, self).__init__()
        self.linear = nn.Linear(latent_dim, latent_dim, bias=True)

        pre_weights = torch.empty([latent_dim, latent_dim])
        nn.init.orthogonal_(pre_weights)
        self.linear.weight.data = pre_weights
        self.linear.bias.data = torch.zeros(latent_dim)

    def forward(self, inputs):
        output = self.linear(inputs)
        return output

# dummy generator, for compatibility with scatterplot code
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, inputs):
        return inputs
