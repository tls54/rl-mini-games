import torch
import torch.nn as nn
import torch.nn.functional as F

BOARD_ROWS = 6
BOARD_COLS = 7


def _build_conv_stack(conv_channels):
    """conv_channels: e.g. [2, 6, 12, 24] -> in_channels, then out_channels per layer."""
    layers = nn.ModuleList()
    for in_ch, out_ch in zip(conv_channels[:-1], conv_channels[1:]):
        layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding='same'))
    return layers


def _build_fc_stack(fc_sizes):
    """fc_sizes: e.g. [1008, 128, 64, 7] -> in_features, then out_features per layer."""
    layers = nn.ModuleList()
    for in_f, out_f in zip(fc_sizes[:-1], fc_sizes[1:]):
        layers.append(nn.Linear(in_f, out_f))
    return layers


class PiTheta(nn.Module):
    def __init__(self, conv_channels=(2, 6, 12, 24), fc_hidden_sizes=(128, 64)):
        super().__init__()
        conv_channels = list(conv_channels)
        self.convs = _build_conv_stack(conv_channels)

        flattened_size = conv_channels[-1] * BOARD_ROWS * BOARD_COLS
        fc_sizes = [flattened_size, *fc_hidden_sizes, BOARD_COLS]
        self.fcs = _build_fc_stack(fc_sizes)

    def forward(self, x):
        board = x
        for conv in self.convs:
            x = F.relu(conv(x))

        x = torch.flatten(x, 1)

        for fc in self.fcs[:-1]:
            x = F.relu(fc(x))
        x = self.fcs[-1](x)

        legal_moves = board[:, 0, 0, :] + board[:, 1, 0, :]

        mask = (legal_moves != 0)

        x = x.masked_fill(mask, float('-inf'))

        return x


class CriticNet(nn.Module):
    def __init__(self, conv_channels=(2, 6, 12, 24), fc_hidden_sizes=(128, 64, 16)):
        super().__init__()
        conv_channels = list(conv_channels)
        self.convs = _build_conv_stack(conv_channels)

        flattened_size = conv_channels[-1] * BOARD_ROWS * BOARD_COLS
        fc_sizes = [flattened_size, *fc_hidden_sizes, 1]
        self.fcs = _build_fc_stack(fc_sizes)

    def forward(self, x):
        for conv in self.convs:
            x = F.relu(conv(x))

        x = torch.flatten(x, 1)

        for fc in self.fcs[:-1]:
            x = F.relu(fc(x))
        x = self.fcs[-1](x)

        return x