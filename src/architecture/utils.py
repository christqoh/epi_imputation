from torch import linspace, Tensor, int32


def get_tensor_dims(tensor: Tensor):
    dims = list(linspace(0, len(tensor.shape) - 1, len(tensor.shape), dtype=int32))
    return dims
