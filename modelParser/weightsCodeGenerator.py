"""
    用来生成权重的C代码
"""

from .nameGenerator import NameGenerator
from .common import tensor2numpy, CTyping
from torch import nn


class WeightsCodeGenerator:
    def __init__(self):
        self.coding = ""

    def __add_coding(self, coding: str):
        if (not coding.endswith("\n")) and (not coding.endswith(";")):
            self.coding += f"{coding};\n"
        elif coding.endswith("\n"):
            self.coding += f"{coding}"
        else:
            self.coding += f"{coding}\n"

    def __add_head_line(self, title):
        self.__add_coding(
            f"\n// ================== {title} ================== //\n")

    def __add_comment(self, comment):
        self.__add_coding(f"// {comment}")

    def __add_variable_array_declaration(self, var_type: str, var_name: str, var_shape: int, var_contents: list[float]):
        var_contents = ", ".join(f"{v:.8f}" for v in var_contents)
        coding = f"{var_type} {var_name}[{var_shape}] = "
        coding += "{" + var_contents + "};"
        self.__add_coding(coding)

    def __add_variable_declaration(self, var_type: str, var_name: str, var_contents: float):
        coding = f"{var_type} {var_name} = {var_contents};"
        self.__add_coding(coding)

    def linear(self, name, linear_layer: nn.Linear):

        weight, bias = linear_layer.weight, linear_layer.bias
        weight, bias = tensor2numpy(weight, bias)

        weight_transposed_var_name = NameGenerator.name_weight_transposed(name)
        bias_transposed_var_name = NameGenerator.name_bias_transposed(name)

        input_row, input_col = weight.shape
        flattened_transposed_weight = weight.T.flatten()

        self.__add_head_line(f"Layer: {name}")
        # Generate C code for transposed weights (1D array)
        self.__add_comment(
            f"Transposed weights for layer: {name} @{input_col}x{input_row}")
        self.__add_variable_array_declaration(
            CTyping.float_t, weight_transposed_var_name, input_col * input_row, flattened_transposed_weight)

        # Generate C code for biases
        self.__add_comment(f"Biases for layer: {name} @{input_row}")
        self.__add_variable_array_declaration(
            CTyping.float_t, bias_transposed_var_name, input_row, bias)

    def conv1d(self, name: str, conv1d_layer: nn.Conv1d):

        weight, bias = conv1d_layer.weight, conv1d_layer.bias

        name_weight = NameGenerator.name_weight(name)
        out_channels, in_channels, kernel_size = weight.shape

        weight = tensor2numpy(weight)

        flattened_transposed_weight = weight.flatten()

        self.__add_head_line(f"Conv1d: {name}")
        # generate c code for weights
        self.__add_comment(
            f"Weight for conv1d: {out_channels}x{in_channels}x{kernel_size} @out_channels x in_channels x kernel_size")
        self.__add_variable_array_declaration(
            CTyping.float_t, name_weight, out_channels * in_channels * kernel_size, flattened_transposed_weight)

        # Generate C code for biases
        if bias is not None:
            self.__add_comment(
                f"Bias for conv1d: {out_channels} @out_channels")
            bias = tensor2numpy(bias)
            name_bias = NameGenerator.name_bias(name)
            self.__add_variable_array_declaration(
                CTyping.float_t, name_bias, out_channels, bias)
        else:
            self.__add_comment("No bias")

    def batchNorm1d(self, name: str, layer: nn.BatchNorm1d):
        self.__add_head_line(f"BatchNorm1d: {name}")
        self.__add_comment(f"BatchNorm1d for layer: {name}")

        name_epsilon = NameGenerator.name_batch_normal1d_epsilon(name)
        name_gamma = NameGenerator.name_batch_normal1d_gamma(name)
        name_beta = NameGenerator.name_batch_normal1d_beta(name)
        name_running_mean = NameGenerator.name_batch_normal1d_running_mean(
            name)
        name_running_var = NameGenerator.name_batch_normal1d_running_var(name)

        epsilon = layer.eps
        gamma = tensor2numpy(layer.weight)
        beta = tensor2numpy(layer.bias)
        running_mean = tensor2numpy(layer.running_mean)
        running_var = tensor2numpy(layer.running_var)

        this_size = gamma.shape[0]

        self.__add_comment(f"epsilon for BatchNorm1d: {epsilon}")
        self.__add_variable_declaration(
            CTyping.float_t, name_epsilon, epsilon
        )
        self.__add_comment(f"gamma for BatchNorm1d: {this_size}")
        self.__add_variable_array_declaration(
            CTyping.float_t, name_gamma, this_size, gamma
        )
        self.__add_comment(f"beta for BatchNorm1d: {this_size}")
        self.__add_variable_array_declaration(
            CTyping.float_t, name_beta, this_size, beta
        )
        self.__add_comment(f"running_mean for BatchNorm1d: {this_size}")
        self.__add_variable_array_declaration(
            CTyping.float_t, name_running_mean, this_size, running_mean
        )
        self.__add_comment(f"running_var for BatchNorm1d: {this_size}")
        self.__add_variable_array_declaration(
            CTyping.float_t, name_running_var, this_size, running_var
        )

    def flatten(self, name):
        self.__add_head_line(f"Flatten: {name}")
        self.__add_comment(f"Flatten for layer: {name}")

    def dropout(self, name):
        self.__add_head_line(f"Dropout: {name}")
        self.__add_comment(f"Dropout for layer: {name}")

    def relu(self, name):
        self.__add_head_line(f"Relu: {name}")
        self.__add_comment(f"Relu for layer: {name}")

    def softMax(self, name):
        self.__add_head_line(f"SoftMax: {name}")
        self.__add_comment(f"SoftMax for layer: {name}")

    def maxPool1d(self, name: str, layer: nn.MaxPool1d):
        self.__add_head_line(f"MaxPool1d: {name}")
        self.__add_comment(f"MaxPool1d for layer: {name}")

    def tanh(self, name):
        self.__add_head_line(f"Tanh: {name}")
        self.__add_comment(f"Tanh for layer: {name}")

    def get_code(self):
        return self.coding
