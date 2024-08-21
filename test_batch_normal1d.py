from modelParser import generate_file_dnn_checkpoint
from torch import nn


class DNN(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.batch_normal_1d = nn.BatchNorm1d(5)

    def forward(self, x):
        x = self.batch_normal_1d(x)
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
