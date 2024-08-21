from modelParser import generate_file_dnn_checkpoint
from torch import nn


class DNN(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.fc1 = nn.Linear(5, 10)
        self.batch_normal_1d = nn.BatchNorm1d(10)
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(10, 20)
        self.relu = nn.ReLU(0.3)
        self.fc5 = nn.Linear(20, 3)
        self.softmax2 = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.batch_normal_1d(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc5(x)
        x = self.softmax2(x)
        return x


dnn_model = DNN()
dnn_model.eval()

batch_size = 1
input_size = 5

generate_file_dnn_checkpoint(
    batch_size,
    input_size,
    dnn_model,
)
