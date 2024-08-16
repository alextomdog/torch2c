from modelParser import ModelParser
from torch import nn
import torch
import subprocess


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

input_ = torch.randn(batch_size, input_size)
output = dnn_model(input_)
print(output)


output_c_file_name = "test_max_pool.c"

model_parser = ModelParser(
    forward_variable_name="result", c_language_filepath="./functions.c")

model_parser.config_single_deep_neral_network(batch_size, input_size)

model_parser.parse_network(dnn_model)

model_parser.save_code(output_c_file_name, input_, output)


def run_c_file(file_name: str):
    print("\nCompiling and running the C file: ", file_name)
    result = subprocess.run(["gcc", file_name, "-o",
                             file_name.split(".")[0]], capture_output=True, text=True)
    if result.returncode == 0:
        execution_result = subprocess.run(
            [f"./{file_name.split('.')[0]}"], capture_output=True, text=True)
        print(execution_result.stdout)
    else:
        print("Compilation failed:", result.stderr)


run_c_file(output_c_file_name)
