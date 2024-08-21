from modelParser import generate_file_dnn_checkpoint
from torch import nn


class Model(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.fc1 = nn.Linear(5, 10)
        self.batch_normal1d = nn.BatchNorm1d(10)
        self.max_pool1 = nn.MaxPool1d(2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(5, 3)

    def forward(self, x):
        x = self.fc1(x)
        x = self.batch_normal1d(x)
        x = self.max_pool1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


dnn_model = Model()
dnn_model.eval()

batch_size = 2
input_size = 5

generate_file_dnn_checkpoint(
    batch_size,
    input_size,
    dnn_model,
)