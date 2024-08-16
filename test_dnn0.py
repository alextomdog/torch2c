from modelParser import ModelParser
from torch import nn
import torch


class DNN(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.fc1 = nn.Linear(5, 10)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(10, 20)
        self.softmax = nn.Softmax(dim=1)
        self.fc5 = nn.Linear(20, 3)
        self.softmax2 = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.softmax(x)
        x = self.fc5(x)
        x = self.softmax2(x)
        return x


dnn_model = DNN()

batch_size = 2
input_size = 5

input_ = torch.randn(batch_size, input_size)
output = dnn_model(input_)
print(output)


model_parser = ModelParser(
    forward_variable_name="result", c_language_filepath="./functions.c")

model_parser.config_single_deep_neral_network(batch_size, input_size)

model_parser.parse_network(dnn_model)

model_parser.save_code("test_dnn0.c", input_, output)
