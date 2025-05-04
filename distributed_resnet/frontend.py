import os
import torch
import torch.distributed.rpc as rpc
from model_parts import DistResNet, Backend
from utils import _call_backend_forward

def run_frontend(rank, world_size, master_addr="localhost", master_port=29500):
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

        backend_rref = rpc.remote("worker0", Backend)
        model = DistResNet(backend_rref)
        model.eval()

        batch_size = 32
        input_tensor = torch.randn(batch_size, 3, 224, 224)
        
        with torch.no_grad():
            out = model(input_tensor)
            
    except Exception as e:
        print(f"Frontend execution failed: {e}")
        raise
    finally:
        if 'backend_rref' in locals():
            backend_rref.delete()
        rpc.shutdown()

if __name__ == "__main__":
    run_frontend(rank=1, world_size=2) 