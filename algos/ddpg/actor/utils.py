import torch
def apply_along_axis(function, x, y, axis: int = 0):
    return torch.stack([
        function(x_i, y_i) for x_i, y_i in zip(torch.unbind(x, dim=axis), torch.unbind(y, dim=axis))
    ], dim=axis)