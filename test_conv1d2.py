from modelParser import generate_file_cnn1d_checkpoint
from torch import nn


class Conv1D(nn.Module):
    def __init__(self):
        super(Conv1D, self).__init__()
        self.conv1 = nn.Conv1d(
            in_channels=2, out_channels=2, kernel_size=3, stride=3)
        self.fl1 = nn.Flatten(start_dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.fl1(x)
        return x


conv1d_model = Conv1D()

batch_size = 1
in_channels = 2
sequence_length = 4

generate_file_cnn1d_checkpoint(
    batch_size,
    in_channels,
    sequence_length,
    conv1d_model
)
