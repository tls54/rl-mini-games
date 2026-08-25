import torch
import torch.nn as nn
import torch.nn.functional as F


class PiTheta(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 6, 3, padding='same')
        self.conv2 = nn.Conv2d(6, 12, 3, padding='same')
        self.conv3 = nn.Conv2d(12, 24, 3, padding='same')
        self.fc1 = nn.Linear(24 * 6 * 7, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 7)

    def forward(self, x):
        board = x
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = torch.flatten(x, 1)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        legal_moves = board[:, 0, 0, :] + board[:, 1, 0, :]

        mask =  (legal_moves != 0)
        
        x = x.masked_fill(mask, float('-inf'))

        return x