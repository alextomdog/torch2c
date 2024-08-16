import torch
import torch.nn as nn

# 输入张量 (batch_size, channels, sequence_length)
input_tensor = torch.tensor([
    [1, -1, 2, -2, 3, -3, 4, -4, 5, -5],
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
], dtype=torch.float32)

# 定义 MaxPool1D 层
maxpool = nn.MaxPool1d(kernel_size=2, stride=2)

# 输入数据需要是 (batch_size, channels, sequence_length) 的格式，这里我们认为 channel = 1
input_tensor = input_tensor.unsqueeze(1)

# 进行 MaxPool1D 操作
output_tensor = maxpool(input_tensor)

# 打印输出结果
print("PyTorch MaxPool1D :")
print(output_tensor.squeeze(1))
