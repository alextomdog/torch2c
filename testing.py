import torch
import torch.nn.functional as F

# 定义输入数据
batch_size = 3
elements_length = 4

input_tensor = torch.tensor([
    [1.0, 2.0, 3.0, 4.0],
    [5.0, 6.0, 7.0, 8.0],
    [9.0, 10.0, 11.0, 12.0]
])

# 转换输入数据为浮点类型
input_tensor = input_tensor.float()

# 定义gamma和beta
gamma = torch.tensor([1.0, 1.0, 1.0, 1.0])
beta = torch.tensor([0.0, 0.0, 0.0, 0.0])

# epsilon值
epsilon = 1e-5

# 计算均值和方差
mean = input_tensor.mean(dim=0)
var = input_tensor.var(dim=0, unbiased=False)

# 批归一化操作
output_tensor = F.batch_norm(
    input_tensor,
    running_mean=mean,
    running_var=var,
    weight=gamma,
    bias=beta,
    training=True,
    momentum=0,
    eps=epsilon
)

print("Normalized output with gamma and beta:")
print(output_tensor)
