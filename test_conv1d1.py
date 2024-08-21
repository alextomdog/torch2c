from modelParser import generate_file_cnn1d_checkpoint
from torch import nn


class Conv1D(nn.Module):
    def __init__(self):
        super(Conv1D, self).__init__()
        self.conv1d_1 = nn.Conv1d(
            in_channels=3, out_channels=5, kernel_size=3, stride=1)
        self.conv1d_2 = nn.Conv1d(
            in_channels=5, out_channels=10, kernel_size=3, stride=1
        )

        self.fl1 = nn.Flatten(start_dim=1)

        self.fc1 = nn.Linear(60, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 5)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1d_1(x)
        x = self.conv1d_2(x)
        x = self.fl1(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x


model = Conv1D()

batch_size = 1
in_channels = 3
seq_len = 10

generate_file_cnn1d_checkpoint(
    batch_size,
    in_channels,
    seq_len,
    model
)
