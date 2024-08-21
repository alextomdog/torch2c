from modelParser import generate_file_cnn1d_checkpoint
from torch import nn


class Model(nn.Module):
    def __init__(self, in_channels):
        super(Model, self).__init__()
        self.conv1 = nn.Conv1d(
            in_channels=in_channels, out_channels=5, kernel_size=3, stride=3)
        self.max_pool1d = nn.MaxPool1d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=20, out_features=32)  # 4 是假定的值
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(32, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.max_pool1d(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x


batch_size = 1
in_channels = 20
sequence_length = 24

model = Model(in_channels)

generate_file_cnn1d_checkpoint(
    batch_size, 
    in_channels,
    sequence_length,
    model
)
