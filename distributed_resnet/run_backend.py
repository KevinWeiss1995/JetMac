from backend import run_backend

if __name__ == "__main__":
    run_backend(
        rank=0, 
        world_size=2,
        master_addr="10.0.0.241",
        master_port=29500
    ) 