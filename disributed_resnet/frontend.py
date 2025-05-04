# frontend.py
import os
import torch
import torch.distributed.rpc as rpc
from model_parts import DistResNet, Backend
from utils import _call_backend_forward

def run_frontend(rank, world_size):
    options = rpc.TensorPipeRpcBackendOptions(
        num_worker_threads=8,
        rpc_timeout=300,
        init_method="tcp://<10.0.0.5>:29500",  # use same Jetson IP here
    )
    rpc.init_rpc("worker1", rank=rank, world_size=world_size, rpc_backend_options=options)

    # Remote reference to backend on Jetson
    backend_rref = rpc.remote("worker0", Backend)

    model = DistResNet(backend_rref)
    model.eval()

    # Sample test input
    input_tensor = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        out = model(input_tensor)
        print("Output:", out)

    rpc.shutdown()

if __name__ == "__main__":
    run_frontend(rank=1, world_size=2)
