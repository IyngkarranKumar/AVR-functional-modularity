import torch
from torch import nn
import torch.nn.functional as F

# helper functions

def default(val, default_val):
    return val if val is not None else default_val

def expand_dim(t, dim, k):
    t = t.unsqueeze(dim)
    expand_shape = [-1] * len(t.shape)
    expand_shape[dim] = k
    return t.expand(*expand_shape)

# simple MLP with ReLU activation

class MLP(nn.Module):
    def __init__(self, *dims, activation = None):
        super().__init__()
        assert len(dims) > 2, 'must have at least 3 dimensions, for dimension in and dimension out'
        activation = default(activation, nn.ReLU)

        layers = []
        pairs = list(zip(dims[:-1], dims[1:]))

        for ind, (dim_in, dim_out) in enumerate(pairs):
            is_last = ind >= (len(pairs) - 1)
            layers.append(nn.Linear(dim_in, dim_out))
            if not is_last:
                layers.append(activation())

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# the feedforward residual block mentioned in the paper
# used after extracting the visual features, as well as post-extraction of attribute information

class FeedForwardResidual(nn.Module):
    def __init__(self, dim, mult = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.LayerNorm(dim * mult),
            nn.ReLU(inplace = True),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        return x + self.net(x)

# convolutional net
# todo, make customizable and add Evonorm for batch independent normalization

class ConvNet(nn.Module):
    def __init__(self, image_size, chans, output_dim):
        super().__init__()

        num_conv_layers = len(chans) - 1
        conv_output_size = image_size // (2 ** num_conv_layers)

        convolutions = []
        channel_pairs = list(zip(chans[:-1], chans[1:]))

        for ind, (chan_in, chan_out) in enumerate(channel_pairs):
            is_last = ind >= (len(channel_pairs) - 1)
            convolutions.append(nn.Conv2d(chan_in, chan_out, 3, padding=1, stride=2))
            if not is_last:
                convolutions.append(nn.BatchNorm2d(chan_out))

        self.net = nn.Sequential(
            *convolutions,
            nn.Flatten(1),
            nn.Linear(chans[-1] * (conv_output_size ** 2), output_dim),
            nn.ReLU(inplace=True),
            FeedForwardResidual(output_dim)
        )

    def forward(self, x):
        return self.net(x)

# scattering transform

class ScatteringTransform(nn.Module):
    def __init__(self, dims, heads, activation = None):
        super().__init__()
        assert len(dims) > 2, 'must have at least 3 dimensions, for dimension in, the hidden dimension, and dimension out'

        dim_in, *hidden_sizes, dim_out = dims

        dim_in //= heads
        dim_out //= heads

        self.heads = heads
        self.mlp = MLP(dim_in, *hidden_sizes, dim_out, activation = activation)

    def forward(self, x):
        shape, heads = x.shape, self.heads
        dim = shape[-1]

        assert (dim % heads) == 0, f'the dimension {dim} must be divisible by the number of heads {heads}'

        x = x.reshape(-1, heads, dim // heads)

        x = self.mlp(x)

        return x.reshape(shape)

# main scattering compositional learner class

class SCL(nn.Module):
    def __init__(
        self,
        image_size = 160,
        set_size = 9,
        conv_channels = [1, 16, 16, 32, 32, 32],
        conv_output_dim = 80,
        attr_heads = 10,
        attr_net_hidden_dims = [128],
        rel_heads = 80,
        rel_net_hidden_dims = [64, 23, 5]):

        super().__init__()
        self.vision = ConvNet(image_size, conv_channels, conv_output_dim)

        self.attr_heads = attr_heads
        self.attr_net = ScatteringTransform([conv_output_dim, *attr_net_hidden_dims, conv_output_dim], heads = attr_heads)
        self.ff_residual = FeedForwardResidual(conv_output_dim)

        self.rel_heads = rel_heads
        self.rel_net = MLP(set_size * (conv_output_dim // rel_heads), *rel_net_hidden_dims)

        self.to_logit = nn.Linear(rel_net_hidden_dims[-1] * rel_heads, 1)

    def forward(self, sets):
        b, m, n, c, h, w = sets.shape
        images = sets.view(-1, c, h, w)
        features = self.vision(images)



        attrs = self.attr_net(features)
        attrs = self.ff_residual(attrs)
        attrs = attrs.reshape(b, m, n, self.rel_heads, -1).transpose(-2, -3).flatten(3)

        
        rels = self.rel_net(attrs)
        rels = rels.flatten(2)
        
        logits = self.to_logit(rels).flatten(1)
        return logits

# wrapper for easier training

class SCLTrainingWrapper(nn.Module):
    def __init__(self, scl):
        super().__init__()
        self.scl = scl

    def forward(self, questions, answers):
        answers = answers.unsqueeze(2)
        questions = expand_dim(questions, dim=1, k=8)

        permutations = torch.cat((questions, answers), dim=2)
        return self.scl(permutations)