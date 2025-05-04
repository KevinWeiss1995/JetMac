import os
import torch
import torch.distributed.rpc as rpc
from model_parts import Backend
import time

def run_backend(rank, world_size, master_addr="localhost", master_port=29500):
    print(f"Starting backend with master_addr={master_addr}, master_port={master_port}")
    
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = str(master_port)
    
    options = rpc.TensorPipeRpcBackendOptions(
        num_worker_threads=8,
        rpc_timeout=300,
        init_method=f"tcp://{master_addr}:{master_port}"
    )
    
    try:
        print("Initializing RPC...")
        rpc.init_rpc(
            "worker0",  # Must match what frontend expects 
            rank=rank,
            world_size=world_size,
            rpc_backend_options=options
        )
        print("RPC initialized successfully")
        
        # Keep backend alive
        while True:
            time.sleep(1)
            
    except Exception as e:
        print(f"RPC initialization failed: {e}")
        raise
    finally:
        if rpc.is_initialized():
            print("Shutting down RPC...")
            rpc.shutdown()

if __name__ == "__main__":
    run_backend(rank=0, world_size=2) 