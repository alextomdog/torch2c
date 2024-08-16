"""
用来生成前向传播的代码
"""


from .nameGenerator import NameGenerator
from .common import tensor2numpy, CTyping
from .cFunctions import CFunctions
from torch import nn


class ForwardCodeGenerator:

    def __init__(self) -> None:
        self.coding = ""
        self.batch_size = None
        self.input_size = None
        self.in_channels = None
        self.sequence_length = None

        self.original_input_size = None

        self.forward_variable_name_tag = None
        self.name_forward_variable_input = None

        self.__is_config = False

    def config_single_deep_neral_network(self, batch_size: int, input_size: int, forward_variable_name="x"):
        self.batch_size = batch_size
        self.input_size = input_size
        self.original_input_size = input_size
        self.forward_variable_name_tag = forward_variable_name

        self.name_forward_variable_input = NameGenerator.name_forward_variable(
            self.forward_variable_name_tag, False)

        self.__is_config = True

    def config_conv1d(self, batch_size: int, in_channels: int, sequence_length: int, forward_variable_name="x"):
        self.batch_size = batch_size
        self.in_channels = in_channels
        self.sequence_length = sequence_length
        self.forward_variable_name_tag = forward_variable_name

        self.name_forward_variable_input = NameGenerator.name_forward_variable(
            self.forward_variable_name_tag, False
        )

        self.original_input_size = in_channels * sequence_length
        self.input_size = self.original_input_size

        self.__is_config = True

    def __check_configuration(self):
        if not self.__is_config:
            raise Exception("Please config first")

    def __add_coding(self, coding: str):
        if (not coding.endswith("\n")) and (not coding.endswith(";")):
            self.coding += f"{coding};\n"
        elif coding.endswith("\n"):
            self.coding += f"{coding}"
        else:
            self.coding += f"{coding}\n"

    def __add_comment(self, comment_string):
        self.__add_coding(f"// {comment_string}\n")

    def __add_layer_declaration(self, var_type: str, var_name: str, function_name: str):
        self.__add_coding(f"{var_type} {var_name} = {function_name};\n")

    def __add_free_declaration(self, var_name: str):
        self.__add_coding(f"free({var_name});\n")

    def linear(self, name, layer_linear: nn.Linear):
        self.__check_configuration()

        weight = layer_linear.weight
        bias = layer_linear.bias

        weight, bias = tensor2numpy(weight, bias)
        output_size = weight.T.shape[1]

        name_weight = NameGenerator.name_weight_transposed(name)
        name_bias = NameGenerator.name_bias_transposed(name)
        name_layer = NameGenerator.name_layer(name)

        name_forward_var1 = NameGenerator.name_forward_variable(
            self.forward_variable_name_tag, False)

        name_forward_var2 = NameGenerator.name_forward_variable(
            self.forward_variable_name_tag)

        self.__add_comment(name_layer)
        self.__add_layer_declaration(CTyping.float_pointer_t, name_forward_var2,
                                     CFunctions.linear(self.batch_size, self.input_size, output_size, name_forward_var1, name_weight, name_bias))
        self.__add_free_declaration(name_forward_var1)

        self.batch_size = self.batch_size
        self.input_size = output_size

    def maxPool1d(self, name, layer: nn.MaxPool1d):
        self.__check_configuration()

        batch_size = self.batch_size
        input_size = self.input_size
        kernel_size = layer.kernel_size
        stride = layer.stride
        padding = layer.padding

        output_size = (input_size + 2 * padding - kernel_size) // stride + 1

        name_forward_var1 = NameGenerator.name_forward_variable(
            self.forward_variable_name_tag, False
        )
        name_forward_var2 = NameGenerator.name_forward_variable(
            self.forward_variable_name_tag, True
        )

        self.__add_comment(NameGenerator.name_layer(name))
        self.__add_comment(f"output_size: {output_size}")
        self.__add_layer_declaration(CTyping.float_pointer_t, name_forward_var2,
                                     CFunctions.maxPool1d(
                                         batch_size,
                                         input_size,
                                         kernel_size,
                                         stride,
                                         padding,
                                         name_forward_var1)
                                     )
        self.__add_free_declaration(name_forward_var1)

        self.batch_size = self.batch_size

        self.input_size = output_size

    def relu(self, name):
        self.__check_configuration()

        relu_var_name = NameGenerator.name_relu(name)

        name_forward = NameGenerator.name_forward_variable(
            self.forward_variable_name_tag, False
        )

        self.__add_comment(relu_var_name)
        self.__add_coding(CFunctions.relu(
            self.batch_size, self.input_size, name_forward))

    def softMax(self, name):
        self.__check_configuration()

        name_forward = NameGenerator.name_forward_variable(
            self.forward_variable_name_tag, False
        )
        self.__add_comment(NameGenerator.name_layer(name))
        self.__add_coding(CFunctions.softmax(
            self.batch_size, self.input_size, name_forward))

    def batchNorm1d(self, name, layer: nn.BatchNorm1d):
        name_forward = NameGenerator.name_forward_variable(
            self.forward_variable_name_tag, False
        )
        self.__add_comment(NameGenerator.name_layer(name))

        name_epsilon = NameGenerator.name_batch_normal1d_epsilon(name)
        name_gamma = NameGenerator.name_batch_normal1d_gamma(name)
        name_beta = NameGenerator.name_batch_normal1d_beta(name)
        name_running_mean = NameGenerator.name_batch_normal1d_running_mean(
            name)
        name_running_var = NameGenerator.name_batch_normal1d_running_var(name)
        name_forward = NameGenerator.name_forward_variable(
            self.forward_variable_name_tag, False
        )

        self.__add_coding(CFunctions.batchNorm1d(
            self.batch_size, self.input_size, name_forward, name_epsilon, name_gamma, name_beta, name_running_mean, name_running_var
        ))

    def flatten(self, name):
        self.__check_configuration()
        self.__add_comment(NameGenerator.name_layer(name))

    def dropout(self, name):
        self.__check_configuration()
        self.__add_comment(NameGenerator.name_layer(name))

    def conv1d(self, name: str, layer: nn.Conv1d):
        self.__check_configuration()
        self.__add_comment(NameGenerator.name_layer(name))

        # Extracting layer parameters
        out_channels = layer.out_channels
        # kernel_size is a tuple (for Conv1d it's a 1-tuple)
        kernel_size = layer.kernel_size[0]
        stride = layer.stride[0]  # stride is also a tuple
        padding = layer.padding[0]  # padding is a tuple as well
        # Converting to numpy array for easier handling
        bias = layer.bias

        # Assuming we have some input tensor x with known shape
        # batch_size, in_channels, sequence_length = 1, layer.in_channels, 10  # Example input dimensions

        # adding the forward variable name
        name_forward_var1 = NameGenerator.name_forward_variable(
            self.forward_variable_name_tag, False
        )
        name_forward_var2 = NameGenerator.name_forward_variable(
            self.forward_variable_name_tag, True
        )
        name_weight = NameGenerator.name_weight(name)
        name_bias = NameGenerator.name_bias(name) if bias is not None else None
        name_layer = NameGenerator.name_layer(name)

        # calculating the output sequence length
        out_sequence_length = (self.sequence_length -
                               kernel_size + 2 * padding) // stride + 1

        # adding the function
        self.__add_comment(
            name_layer + f"(batch_size: {self.batch_size}, in_channels: {self.in_channels}, sequence_length: {self.sequence_length})")
        self.__add_comment(
            f"=> @(batch_size: {self.batch_size}, out_channels: {out_channels}, out_sequence_length: {out_sequence_length})")
        function_conv1d = CFunctions.conv1d(
            self.batch_size,
            self.in_channels,
            self.sequence_length,
            name_forward_var1,
            out_channels,
            kernel_size,
            stride,
            padding,
            name_weight,
            name_bias
        )
        self.__add_layer_declaration(
            CTyping.float_pointer_t, name_forward_var2, function_conv1d)
        self.__add_free_declaration(name_forward_var1)

        self.in_channels = out_channels
        self.sequence_length = out_sequence_length
        self.input_size = out_channels * out_sequence_length

    def get_code(self):
        self.__check_configuration()
        name_last_variable = NameGenerator.name_forward_variable(
            self.forward_variable_name_tag, False
        )

        def memory_copy(variable_from, variable_to, length):
            return f"for (int i = 0; i < {length}; i++) {{ {variable_to}[i] = {variable_from}[i]; }}\n"

        prefix = ""
        prefix += "float* forward(float input[], float output[]){\n"
        prefix += f"\tfloat* {self.name_forward_variable_input}=(float*)malloc(sizeof(float)*{self.original_input_size * self.batch_size});\n"
        prefix += memory_copy("input", self.name_forward_variable_input,
                              self.original_input_size * self.batch_size)

        suffix = memory_copy(name_last_variable, "output",
                             self.input_size * self.batch_size)

        suffix += f"\tfree({name_last_variable});\n"
        suffix += "}"

        return prefix + self.coding + suffix
