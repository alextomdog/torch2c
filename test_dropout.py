import subprocess
from modelParser import ModelParser
from torch import nn
import torch


class DNN(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.fc1 = nn.Linear(5, 10)
        self.relu1 = nn.ReLU()
        # Dropout layer with 50% dropout rate
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(10, 20)
        self.dropout2 = nn.Dropout(0.3)
        self.fc5 = nn.Linear(20, 3)
        self.softmax2 = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        x = self.fc5(x)
        x = self.softmax2(x)
        return x


dnn_model = DNN()
dnn_model.eval()

batch_size = 1
input_size = 5

input_ = torch.randn(batch_size, input_size)
output = dnn_model(input_)
print(output)

output_c_file_name = "test_dropout.c"

model_parser = ModelParser(
    forward_variable_name="result", c_language_filepath="./functions.c")

model_parser.config_single_deep_neral_network(batch_size, input_size)

model_parser.parse_network(dnn_model)

model_parser.save_code(output_c_file_name, input_, output)


def run_c_file(file_name):
    print("\n\nCompiling and running the C file: ", file_name)
    result = subprocess.run(["gcc", file_name, "-o",
                             "test_batch_normal1d"], capture_output=True, text=True)
    if result.returncode == 0:
        execution_result = subprocess.run(
            ["./test_batch_normal1d"], capture_output=True, text=True)
        print(execution_result.stdout)
    else:
        print("Compilation failed:", result.stderr)


run_c_file(output_c_file_name)
