from modelParser import generate_file_cnn1d_checkpoint
from torch import nn


class Conv1D(nn.Module):
    def __init__(self):
        super(Conv1D, self).__init__()
        self.conv1 = nn.Conv1d(
            in_channels=2, out_channels=2, kernel_size=3, stride=1)
        self.fl1 = nn.Flatten(start_dim=1)
        self.fc1 = nn.Linear(4, 10)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(10, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.fl1(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x


conv1d_model = Conv1D()

in_channels = 2
batch_size = 2
sequence_length = 4

generate_file_cnn1d_checkpoint(
    batch_size,
    in_channels,
    sequence_length,
    conv1d_model
)
