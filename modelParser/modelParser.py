"""
用来解析pytorch模型的代码
"""

from .forwardCodeGenerator import ForwardCodeGenerator
from .weightsCodeGenerator import WeightsCodeGenerator
from .cLanguageSection import CLanguageSection
from .nameGenerator import NameGenerator
from .cFunctions import CFunctionsName
import torch
from torch import nn


class ModelParser:

    def __init__(self, forward_variable_name="x", c_language_filepath: str | None = None):
        self.forward_variable_name_tag = forward_variable_name
        self.c_language_filepath = c_language_filepath

        self.forward_code_generator = None
        self.weights_code_generator = None
        self.c_language_section = None

        self.__is_config = False

    def __check_configuration(self):
        if not self.__is_config:
            raise Exception("Please config first!")

    def config_single_deep_neral_network(self, batch_size: int, input_size: int):
        # 配置forward区域代码生成器
        self.forward_code_generator = ForwardCodeGenerator()
        self.forward_code_generator.config_single_deep_neral_network(
            batch_size, input_size, self.forward_variable_name_tag
        )
        # 配置weights区域代码生成器
        self.weights_code_generator = WeightsCodeGenerator()

        # 配置c语言附加文件区域
        if self.c_language_filepath is not None:
            self.c_language_section = CLanguageSection(
                self.c_language_filepath)
            self.c_language_section.config_linear_as_first_layer(
                batch_size, input_size)

        self.__is_config = True

    def config_conv1d_neral_network(self, batch_size: int, in_channels: int, sequence_length: int):
        # 配置forward区域代码生成器
        self.forward_code_generator = ForwardCodeGenerator()
        self.forward_code_generator.config_conv1d(
            batch_size, in_channels, sequence_length, self.forward_variable_name_tag)

        # 配置weights区域代码生成器
        self.weights_code_generator = WeightsCodeGenerator()

        # 配置c语言附加文件区域
        if self.c_language_filepath is not None:
            self.c_language_section = CLanguageSection(
                self.c_language_filepath
            )
            self.c_language_section.config_conv1d_as_first_layer(
                batch_size, in_channels, sequence_length)

        self.__is_config = True

    def parse_network(self, model) -> bool:
        self.__check_configuration()
        NameGenerator.initialization()

        for name, layer in model.named_children():
            if isinstance(layer, nn.Linear):
                self.weights_code_generator.linear(name, layer)
                self.forward_code_generator.linear(name, layer)
                if self.c_language_section is not None:
                    self.c_language_section.adding_callback_function_from_name(
                        CFunctionsName.linear)

            elif isinstance(layer, nn.ReLU):
                self.weights_code_generator.relu(name)
                self.forward_code_generator.relu(name)
                if self.c_language_section is not None:
                    self.c_language_section.adding_callback_function_from_name(
                        CFunctionsName.relu)

            elif isinstance(layer, nn.Softmax):
                self.weights_code_generator.softMax(name)
                self.forward_code_generator.softMax(name)
                if self.c_language_section is not None:
                    self.c_language_section.adding_callback_function_from_name(
                        CFunctionsName.softmax)

            elif isinstance(layer, nn.Conv1d):
                self.weights_code_generator.conv1d(name, layer)
                self.forward_code_generator.conv1d(name, layer)
                if self.c_language_section is not None:
                    self.c_language_section.adding_callback_function_from_name(
                        CFunctionsName.conv1d)

            elif isinstance(layer, nn.Flatten):
                self.forward_code_generator.flatten(name)
                self.weights_code_generator.flatten(name)
                if self.c_language_section is not None:
                    self.c_language_section.adding_callback_function_from_name(
                        CFunctionsName.flatten)

            elif isinstance(layer, nn.Dropout):
                self.forward_code_generator.dropout(name)
                self.weights_code_generator.dropout(name)
                if self.c_language_section is not None:
                    self.c_language_section.adding_callback_function_from_name(
                        CFunctionsName.dropout)

            elif isinstance(layer, nn.BatchNorm1d):
                self.weights_code_generator.batchNorm1d(name, layer)
                self.forward_code_generator.batchNorm1d(name, layer)
                if self.c_language_section is not None:
                    self.c_language_section.adding_callback_function_from_name(
                        CFunctionsName.batchNorm1d)

            elif isinstance(layer, nn.MaxPool1d):
                self.weights_code_generator.maxPool1d(name, layer)
                self.forward_code_generator.maxPool1d(name, layer)
                if self.c_language_section is not None:
                    self.c_language_section.adding_callback_function_from_name(
                        CFunctionsName.maxPool1d
                    )

        return True

    def save_code(self,
                  filename="model",
                  prepared_input_to_model_for_generate__main__: torch.Tensor | None = None,
                  gotten_output_from_model_for_generate__main__: torch.Tensor | None = None):
        self.__check_configuration()

        body_filename = filename + ".c"
        header_filename = filename + ".h"

        coding = self.get_code(prepared_input_to_model_for_generate__main__,
                               gotten_output_from_model_for_generate__main__)

        with open(body_filename, "w", encoding="utf-8") as f:
            f.write(coding)

        with open(header_filename, "w", encoding="utf-8") as f:
            f.write(self.get_header_code(filename))

    def get_code(self,
                 prepared_input_to_model_for_generate__main__: torch.Tensor | None = None,
                 gotten_output_from_model_for_generate__main__: torch.Tensor | None = None):
        self.__check_configuration()

        coding = ""

        # 如果c语言文件区域不为空，则添加头文件
        if self.c_language_section is not None:
            coding += self.c_language_section.get_libs()

        # 添加权重代码
        coding += f"{self.weights_code_generator.get_code()}\n"

        # 如果c语言文件区域不为空，则添加模型调用函数代码
        if self.c_language_section is not None:
            coding += f"{self.c_language_section.get_code()}\n"

        # 添加向前传播forward代码
        coding += f"{self.forward_code_generator.get_code()}\n"

        # 如果c语言文件区域不为空，则添加main函数
        if self.c_language_section is not None:
            if prepared_input_to_model_for_generate__main__ is not None and gotten_output_from_model_for_generate__main__ is not None:
                coding += self.c_language_section.generate_main_func(
                    prepared_input_to_model_for_generate__main__,
                    gotten_output_from_model_for_generate__main__)

        return coding

    def get_header_code(self, define_name="model"):
        return self.c_language_section.get_header_code(define_name)
