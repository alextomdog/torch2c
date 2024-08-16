import subprocess
import os
from modelParser import ModelParser
from torch import nn
import torch


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

input_ = torch.randn(batch_size, input_size)

output = dnn_model(input_)
print(output)
# print("eps: ", dnn_model.batch_normal_1d.eps)
# print("beta: ", dnn_model.batch_normal_1d.weight)
# print("gamma: ", dnn_model.batch_normal_1d.bias)
# print("mean: ", dnn_model.batch_normal_1d.running_mean)
# print("var: ", dnn_model.batch_normal_1d.running_var)

output_c_file_name = "test_batch_normal1d.c"

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
