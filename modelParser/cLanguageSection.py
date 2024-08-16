"""
用来生成c语言头文件和提取相关函数以及main函数的类
"""
from .cFunctions import CFunctionsName
import torch


class CLanguageSection:

    def __init__(self, c_language_file_path="./functions.c") -> None:
        self.input_size = None
        self.batch_size = None

        self.c_language_file_path = c_language_file_path

        self.coding = ""

        self.cache_c_functions = []

    def config_linear_as_first_layer(self, batch_size: int, input_size: int):
        self.input_size = input_size
        self.batch_size = batch_size

    def config_conv1d_as_first_layer(self, batch_size: int, in_channels: int, sequence_length: int):
        self.input_size = in_channels*sequence_length
        self.batch_size = batch_size

    def get_libs(self) -> str:
        libs = [
            "stdio.h",
            "stdlib.h",
            "math.h",
            "stdbool.h",
            "float.h"
        ]
        lib_string = ""
        for lib_str in libs:
            lib_string += f"#include <{lib_str}>\n"
        return lib_string

    def generate_main_func(self,
                           prepared_input_to_model: torch.Tensor,
                           gotten_output_from_model: torch.Tensor):
        prepared_input_to_model = prepared_input_to_model.flatten(
            start_dim=0).detach().numpy().tolist()
        prepared_input_to_model = [str(i) for i in prepared_input_to_model]

        gotten_output_from_model = gotten_output_from_model.flatten(
            start_dim=0).detach().numpy().tolist()
        output_final_size = len(gotten_output_from_model)

        main_func_coding = "int main(){\n"
        input_string_like_array = ",".join(prepared_input_to_model)
        main_func_coding += f"float input[{self.batch_size * self.input_size}] = {{ {input_string_like_array} }};\n"
        main_func_coding += f"float output[{output_final_size}];\n"
        main_func_coding += f"forward(input, output);\n"
        main_func_coding += f"for (int i = 0; i < {output_final_size}; i++){{ printf(\"%f  \", output[i]); \n }}\n"
        main_func_coding += "return 0;\n"
        main_func_coding += "}"
        return main_func_coding

    def adding_callback_function_from_name(self, searching_function_name: str) -> None:
        # 如果函数在白名单中，则不添加
        for white_function_name in CFunctionsName.get_white_list():
            if white_function_name == searching_function_name:
                return

        # 如果已经存在，则不再添加
        if self.cache_c_functions.count(searching_function_name) > 0:
            return

        file_content = ""

        # 读取文件内容
        with open(self.c_language_file_path, mode="r", encoding="utf-8") as f:
            file_content = f.read()

        stack = []

        searching_function_name_start_index = file_content.find(
            searching_function_name)
        if searching_function_name_start_index == -1:
            raise Exception(f"Function {searching_function_name} not found")

        # 找到函数开始的位置
        while True:
            # 如果是开始头第一个函数
            if searching_function_name_start_index == -1:
                searching_function_name_start_index = 0
                break
            # 如果是中间的函数
            if file_content[searching_function_name_start_index] == "\n":
                searching_function_name_start_index += 1
                break
            elif file_content[searching_function_name_start_index] == "}":
                searching_function_name_start_index += 1
                break
            searching_function_name_start_index -= 1

        searching_result_function = ""
        # 找到函数开始的section，同时保留函数的声明部分
        while True:
            searching_result_function += file_content[searching_function_name_start_index]
            searching_function_name_start_index += 1
            if file_content[searching_function_name_start_index] == '{':
                stack.append('{')
                break

        # 找到函数结束的section
        while True:
            searching_result_function += file_content[searching_function_name_start_index]
            searching_function_name_start_index += 1
            if file_content[searching_function_name_start_index] == '{':
                stack.append('{')
            elif file_content[searching_function_name_start_index] == '}':
                stack.pop()
                if len(stack) == 0:
                    searching_result_function += file_content[searching_function_name_start_index]
                    break

        # 将函数名字添加到缓存列表中
        self.cache_c_functions.append(searching_function_name)

        # 将函数添加到coding中
        self.coding += searching_result_function + "\n"

    def get_code(self) -> str:
        return self.coding
