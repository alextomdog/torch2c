from modelParser import generate_file_cnn1d_checkpoint
from torch import nn


class Conv1D(nn.Module):
    def __init__(self):
        super(Conv1D, self).__init__()
        self.conv1 = nn.Conv1d(
            in_channels=2, out_channels=5, kernel_size=3, stride=3)
        self.conv2 = nn.Conv1d(
            in_channels=5, out_channels=1, kernel_size=1, stride=2)
        self.conv3 = nn.Conv1d(
            in_channels=1, out_channels=1, kernel_size=1, stride=1)
        self.fl1 = nn.Flatten()
        self.fc1 = nn.Linear(8, 16)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(16, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.fl1(x)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x


conv1d_model = Conv1D()

batch_size = 1
in_channels = 2
sequence_length = 50

generate_file_cnn1d_checkpoint(
    batch_size,
    in_channels,
    sequence_length,
    conv1d_model
)
