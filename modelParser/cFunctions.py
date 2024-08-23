"""
用来做c语言和python语言接口的函数
"""


class CFunctionsName:
    relu = "Relu"
    softmax = "SoftMax"
    conv1d = "Conv1d"
    linear = "Linear"
    batchNorm1d = "BatchNorm1d"
    maxPool1d = "MaxPool1D"
    tanh = "Tanh"
    LeakyReLU = "LeakyReLU"
    sigmoid = "sigmoid"

    # not to be used
    flatten = "Flatten"
    dropout = "Dropout"

    @staticmethod
    def get_white_list() -> list[str]:
        return [CFunctionsName.flatten, CFunctionsName.dropout]


class CFunctions:

    @staticmethod
    def leakyReLU(batch_size, total_elements_per_batch, input_variable_name, alpha):
        # void LeakyReLU(int batch_size, int elements_length, float *input, float alpha)
        return f"{CFunctionsName.LeakyReLU}({batch_size},{total_elements_per_batch},{input_variable_name},{alpha})"

    @staticmethod
    def relu(batch_size, total_elements_per_batch, input_variable_name):
        return f"{CFunctionsName.relu}({batch_size},{total_elements_per_batch},{input_variable_name})"

    @staticmethod
    def sigmoid(batch_size, total_elements_per_batch, input_variable_name):
        # void sigmoid(int batch_size, int elements_length, float *input)
        return f"{CFunctionsName.sigmoid}({batch_size},{total_elements_per_batch},{input_variable_name})"

    @staticmethod
    def tanh(batch_size, total_elements_per_batch, input_variable_name):
        # void tanh(int batch_size, int elements_length, float *input)
        return f"{CFunctionsName.tanh}({batch_size},{total_elements_per_batch},{input_variable_name})"

    @staticmethod
    def softmax(batch_size, total_elements_per_batch, input_variable_name):
        return f"{CFunctionsName.softmax}({batch_size},{total_elements_per_batch},{input_variable_name})"

    @staticmethod
    def maxPool1d(batch_size: int,
                  total_elements_per_batch: int,
                  pool_size: int,
                  stride: int,
                  padding: int,
                  input_variable_name: str):
        # void MaxPool1D(int batch_size, int elements_length, int pool_size, int stride, int padding, float *input, float *output)
        return f"{CFunctionsName.maxPool1d}({batch_size},{total_elements_per_batch},{pool_size},{stride},{padding},{input_variable_name})"

    @staticmethod
    def batchNorm1d(batch_size,
                    total_elements_per_batch,
                    input_variable_name,
                    epsilon,
                    gamma: list[float],
                    beta: list[float],
                    running_mean: list[float],
                    running_var: list[float]):
        # void BatchNorm1d(int batch_size, int elements_length, float *input, float epsilon, float *gamma, float *beta)
        return f"{CFunctionsName.batchNorm1d}({batch_size},{total_elements_per_batch},{input_variable_name},{epsilon},{gamma},{beta}, {running_mean},{running_var})"

    @staticmethod
    def linear(batch_size, input_size, output_size, name_forward, name_weight, name_bias):
        return f"{CFunctionsName.linear}({batch_size},{input_size},{output_size},{name_forward},{name_weight},{name_bias})"

    @staticmethod
    def conv1d(batch_size: int,
               in_channels: int,
               sequence_length: int,
               input_variable_name: str,
               out_channels: int,
               kernel_size: int,
               stride: int,
               padding: int,
               weights: str,
               bias: str | None):
        if bias is None:
            bias = "NULL"
            use_bias = "false"
        else:
            use_bias = "true"
        # float *Conv1d(int batch_size, int in_channels, int sequence_length, float *input, int out_channels, int kernel_size, int stride, int padding, float *weights, float *bias, bool use_bias);
        return f"{CFunctionsName.conv1d}({batch_size},{in_channels},{sequence_length}, {input_variable_name}, {out_channels}, {kernel_size}, {stride}, {padding}, {weights}, {bias}, {use_bias})"
