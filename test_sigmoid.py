from modelParser import generate_file_dnn_checkpoint
from torch import nn


class DNN(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.fc1 = nn.Linear(5, 10)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(10, 20)
        self.sigmoid = nn.Sigmoid()
        self.fc5 = nn.Linear(20, 3)
        self.tanh = nn.Tanh()
        self.fc6 = nn.Linear(3, 2)
        self.softmax2 = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        x = self.fc5(x)
        x = self.tanh(x)
        x = self.fc6(x)
        x = self.softmax2(x)
        return x


dnn_model = DNN()

batch_size = 1
input_size = 5

generate_file_dnn_checkpoint(
    batch_size,
    input_size,
    dnn_model,
)
