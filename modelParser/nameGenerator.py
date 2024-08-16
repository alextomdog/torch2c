"""
用来生成c语言变量名的类
"""


class NameGenerator:
    forward_var_name_index = 0

    @staticmethod
    def initialization():
        NameGenerator.forward_var_name_index = 0

    @staticmethod
    def name_forward_variable(name, is_change=True):
        if is_change:
            NameGenerator.forward_var_name_index += 1
        return f"{name}_{NameGenerator.forward_var_name_index}"

    @staticmethod
    def name_layer(name):
        return f"{name}_layer"

    @staticmethod
    def name_weight(name):
        return f"{name}_weights"

    @staticmethod
    def name_bias(name):
        return f"{name}_bias"

    @staticmethod
    def name_weight_transposed(name):
        return f"{name}_weight_transposed"

    @staticmethod
    def name_bias_transposed(name):
        return f"{name}_bias_transposed"

    @staticmethod
    def name_relu(name):
        return f"{name}_relu"

    @staticmethod
    def name_batch_normal1d_epsilon(name):
        return f"{name}_batch_normal1d_epsilon"

    @staticmethod
    def name_batch_normal1d_gamma(name):
        return f"{name}_batch_normal1d_gamma"

    @staticmethod
    def name_batch_normal1d_beta(name):
        return f"{name}_batch_normal1d_beta"

    @staticmethod
    def name_batch_normal1d_running_mean(name):
        return f"{name}_batch_normal1d_running_mean"

    @staticmethod
    def name_batch_normal1d_running_var(name):
        return f"{name}_batch_normal1d_running_var"
