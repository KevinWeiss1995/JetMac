from frontend import run_frontend

if __name__ == "__main__":
    run_frontend(
        rank=1, 
        world_size=2,
        master_addr="10.0.0.241",
        master_port=29500
    ) 