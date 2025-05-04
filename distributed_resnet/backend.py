import os
import torch
import torch.distributed.rpc as rpc
from model_parts import Backend

def run_backend(rank, world_size, master_addr="localhost", master_port=29500):
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = str(master_port)
    
    options = rpc.TensorPipeRpcBackendOptions(
        num_worker_threads=8,
        rpc_timeout=300,
        init_method=f"tcp://{master_addr}:{master_port}"
    )
    
    try:
        rpc.init_rpc(
            f"worker{rank}", 
            rank=rank,
            world_size=world_size,
            rpc_backend_options=options
        )
        
        rpc._wait_all_workers()
    except Exception as e:
        print(f"RPC initialization failed: {e}")
        raise
    finally:
        rpc.shutdown()

if __name__ == "__main__":
    run_backend(rank=0, world_size=2) 