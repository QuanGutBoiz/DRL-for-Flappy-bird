import torch
import torch.nn as nn
import torch.nn.functional as F
class DuelingDQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DuelingDQN, self).__init__()
        self.feature_layer = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(self.feature_size(input_shape), 512),
            nn.ReLU()
        )
        self.value_stream = nn.Linear(512, 1)
        self.advantage_stream = nn.Linear(512, n_actions)

    def feature_size(self, input_shape):
        return self.feature_layer(torch.zeros(1, *input_shape)).view(1, -1).size(1)

    def forward(self, x):
        x = self.feature_layer(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layer(x)
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        return value + (advantage - advantage.mean())