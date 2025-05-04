import torch

def _call_backend_forward(backend_rref, x):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = x.to(device, non_blocking=True)
    try:
        result = backend_rref.to_here()(x)
        return result.to("cpu") if device.type == "cuda" else result
    finally:
        del x 