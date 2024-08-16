from modelParser import ModelParser
import torch
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
input_ = torch.randn(batch_size, in_channels, sequence_length)
output = conv1d_model(input_)
print(output)

parser = ModelParser(forward_variable_name="result",
                     c_language_filepath="./functions.c")

parser.config_conv1d_neral_network(batch_size, in_channels, sequence_length)

parser.parse_network(conv1d_model)

parser.save_code("test_conv1d2.c", input_, output)
