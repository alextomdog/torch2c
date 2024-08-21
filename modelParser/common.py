from torch import Tensor
import numpy as np


def tensor2numpy(*tensors: Tensor) -> list:
    result = [tensor.detach().numpy().astype(np.float16) for tensor in tensors]
    if len(result) == 1:
        return result[0]
    return result


class CTyping:
    float_t = "float"
    int_t = "int"
    bool_t = "bool"
    float_pointer_t = "float*"
    void_t = "void"
