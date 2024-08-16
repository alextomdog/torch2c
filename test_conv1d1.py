from modelParser import ModelParser
import torch
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

input_ = torch.randn(batch_size, in_channels, seq_len)
output = model(input_)

print(output)


model_parser = ModelParser(
    forward_variable_name="result", c_language_filepath="./functions.c")

model_parser.config_conv1d_neral_network(batch_size, in_channels, seq_len)

model_parser.parse_network(model)

model_parser.save_code("./test_conv1d1.c", input_, output)
