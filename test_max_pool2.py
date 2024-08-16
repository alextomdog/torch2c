import subprocess
from modelParser import ModelParser
import torch
from torch import nn


class Model(nn.Module):
    def __init__(self, in_channels):
        super(Model, self).__init__()
        self.conv1 = nn.Conv1d(
            in_channels=in_channels, out_channels=5, kernel_size=3, stride=3)
        self.max_pool1d = nn.MaxPool1d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=20, out_features=32)  # 4 是假定的值
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(32, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.max_pool1d(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x


batch_size = 1
in_channels = 20
sequence_length = 24

# 测试模型结构
model = Model(in_channels)
model.eval()
# 假设输入尺寸为 (batch_size, channels, length)
input_ = torch.randn(batch_size, in_channels, sequence_length)
output = model(input_)
print(output)

output_c_file_name = "test_max_pool2.c"

model_parser = ModelParser(
    forward_variable_name="result", c_language_filepath="./functions.c")

model_parser.config_conv1d_neral_network(
    batch_size, in_channels, sequence_length)

model_parser.parse_network(model)

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
