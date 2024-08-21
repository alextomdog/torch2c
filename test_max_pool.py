from modelParser import generate_file_dnn_checkpoint
from torch import nn


class DNN(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.fc1 = nn.Linear(5, 10)
        self.max_pool = nn.MaxPool1d(2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.max_pool(x)
        return x


dnn_model = DNN()

batch_size = 2
input_size = 5

generate_file_dnn_checkpoint(
    batch_size,
    input_size,
    dnn_model,
)