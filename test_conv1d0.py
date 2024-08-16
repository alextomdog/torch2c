from modelParser import ModelParser
import torch
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
input_ = torch.randn(batch_size, in_channels, sequence_length)
output = conv1d_model(input_)
print(output)

parser = ModelParser(forward_variable_name="result",
                     c_language_filepath="./functions.c")

parser.config_conv1d_neral_network(batch_size, in_channels, sequence_length)

parser.parse_network(conv1d_model)

parser.save_code("test_conv1d0.c", input_, output)
