from .modelParser import ModelParser
import torch
import subprocess

__all__ = ["generate_file_cnn1d_checkpoint"]


def run_c_file(file_name: str):
    print("\nCompiling and running the C file: ", file_name)
    result = subprocess.run(["gcc", file_name, "-o",
                             file_name.split(".")[0]], capture_output=True, text=True)
    if result.returncode == 0:
        execution_result = subprocess.run(
            [f"./{file_name.split('.')[0]}"], capture_output=True, text=True)
        print(execution_result.stdout)
        print()
    else:
        print("Compilation failed:", result.stderr)


def generate_file_cnn1d_checkpoint(
    batch_size: int,
    in_channels: int,
    sequence_length: int,
    model: torch.nn.Module,
    output_filename: str = "model",
    c_language_filepath: str = "./functions.c",
    forward_variable_name: str = "result",
):

    input_ = torch.randn(batch_size, in_channels, sequence_length)
    model.eval()
    output = model(input_)
    output_list = output.squeeze().flatten().detach().numpy().tolist()
    print(f"\noriginal model output:")
    for i in output_list:
        print(round(i, 6), end="  ")
    print()

    model_parser = ModelParser(
        forward_variable_name=forward_variable_name, c_language_filepath=c_language_filepath
    )

    model_parser.config_conv1d_neral_network(
        batch_size, in_channels, sequence_length
    )

    model_parser.parse_network(model)

    model_parser.save_code(
        output_filename, input_, output)

    # 运行c文件
    output_c_filename = output_filename + ".c"
    run_c_file(output_c_filename)


def generate_file_dnn_checkpoint(
    batch_size: int,
    input_size: int,
    model: torch.nn.Module,
    output_filename: str = "model",
    c_language_filepath: str = "./functions.c",
    forward_variable_name: str = "result",
):
    input_ = torch.randn(batch_size, input_size)
    model.eval()

    output = model(input_)
    output_list = output.squeeze().flatten().detach().numpy().tolist()
    print(f"\noriginal model output:")
    for i in output_list:
        print(round(i, 6), end="  ")
    print()

    model_parser = ModelParser(
        forward_variable_name=forward_variable_name, c_language_filepath=c_language_filepath
    )

    model_parser.config_single_deep_neral_network(batch_size, input_size)

    model_parser.parse_network(model)

    model_parser.save_code(output_filename, input_, output)

    # 运行c文件
    output_c_filename = output_filename + ".c"
    run_c_file(output_c_filename)
