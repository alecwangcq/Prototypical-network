import torch.nn as nn
import torch


def euclidean_dist(x, y):
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2))


class ProtoDisentangle(nn.Module):
    def __init__(self, in_dim, ind_dim, den_dim):
        super(ProtoDisentangle, self).__init__()
        self.encoder_ind = nn.Sequential(nn.Dropout(0.5),
                                         nn.Linear(in_dim, in_dim),
                                         nn.Dropout(0.5),
                                         nn.LeakyReLU(),
                                         nn.Linear(in_dim, ind_dim))
        self.encoder_den = nn.Sequential(nn.Dropout(0.5),
                                         nn.Linear(in_dim, in_dim),
                                         nn.Dropout(0.5),
                                         nn.LeakyReLU(),
                                         nn.Linear(in_dim, den_dim))

        self.decoder = nn.Sequential(nn.Dropout(0.5),
                                     nn.Linear(ind_dim+den_dim, in_dim),
                                     nn.Dropout(0.5),
                                     nn.LeakyReLU(),
                                     nn.Linear(in_dim, in_dim))

    def encoding(self, x):
        return self.encoder_ind(x), self.encoder_den(x)

    def decoding(self, x):
        return self.decoder(x)


class ProtoConvNet(nn.Module):
    def __init__(self, x_dim, hid_dim, z_dim):
        super(ProtoConvNet, self).__init__()
        self.net = nn.Sequential(conv_block(x_dim, hid_dim),
                                 conv_block(hid_dim, hid_dim),
                                 conv_block(hid_dim, hid_dim),
                                 conv_block(hid_dim, z_dim),
                                 Flatten())

    def forward(self, x):
        return self.net(x)


class ProtoRadiusPredictor(nn.Module):
    def __init__(self, in_dim):
        super(ProtoRadiusPredictor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.ReLU(),
            nn.Linear(in_dim // 2, 1))
        self.init_weights()

    def init_weights(self):
        self.net[2].weight.data.uniform_(0, 1.0)
        self.net[2].bias.data.fill_(0.0)

    def forward(self, x):
        r = self.net(x)
        r = torch.log(1.0 + torch.exp(r))
        return r

