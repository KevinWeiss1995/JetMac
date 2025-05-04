import os
import torch
import torch.distributed.rpc as rpc
from model_parts import Backend

def run_backend(rank, world_size):
    options = rpc.TensorPipeRpcBackendOptions(
        num_worker_threads=8,
        rpc_timeout=300,
        init_method="tcp://<10.0.0.5>:29500",  
    )
    rpc.init_rpc("worker0", rank=rank, world_size=world_size, rpc_backend_options=options)
    print("Backend ready. Waiting for frontend to invoke.")
    rpc.shutdown()

if __name__ == "__main__":
    run_backend(rank=0, world_size=2)
