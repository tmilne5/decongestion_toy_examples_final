import torch


def to_polar(data):
    """
    Takes a minibatch of 2-D data, assumes it is (r,theta) values, and
    transforms the data to polar coordinates.
    """
    xs = data[:, 0] * torch.cos(data[:, 1])
    ys = data[:, 0] * torch.sin(data[:, 1])

    return torch.stack([xs, ys], dim=1).detach().clone()
