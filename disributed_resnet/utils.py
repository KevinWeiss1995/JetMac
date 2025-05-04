# utils.py
import torch

def _call_backend_forward(backend_rref, x):
    x = x.cuda(non_blocking=True)
    return backend_rref.to_here()(x)
