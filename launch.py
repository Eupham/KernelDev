"""
Multi-process training launcher script.

This script launches multiple instances of a training script for distributed training
on a single node. It sets the necessary environment variables for PyTorch's
distributed communication (MASTER_ADDR, MASTER_PORT, WORLD_SIZE, RANK, LOCAL_RANK).

Usage:
    python launch.py --nproc_per_node <num_gpus> <path_to_training_script.py> [args_for_training_script...]

Example:
    python launch.py --nproc_per_node 2 entry.py --epochs 10 --batch_size 32

This will launch 2 processes of `entry.py`, each with the appropriate environment
variables set for distributed training. `entry.py` will receive `--epochs 10` and
`--batch_size 32` as its arguments.
"""
import sys
import subprocess
import os
import socket
import argparse

def find_free_port() -> str:
    """
    Finds an available port on the system.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(('localhost', 0))
    port = sock.getsockname()[1]
    sock.close()
    return str(port)

def main():
    parser = argparse.ArgumentParser(
        description="PyTorch Distributed Training Launcher for single-node execution."
    )
    parser.add_argument(
        "--nproc_per_node",
        type=int,
        required=True,
        help="Number of processes to launch per node (typically the number of GPUs)."
    )
    parser.add_argument(
        "training_script",
        type=str,
        help="Path to the training script to execute (e.g., entry.py)."
    )
    parser.add_argument(
        "training_script_args",
        nargs=argparse.REMAINDER,
        help="Arguments to pass to the training script."
    )

    args = parser.parse_args()

    if args.nproc_per_node <= 0:
        print("Error: --nproc_per_node must be a positive integer.")
        sys.exit(1)

    MASTER_ADDR = '127.0.0.1'
    MASTER_PORT = find_free_port()
    WORLD_SIZE = args.nproc_per_node

    print(f"Master Addr: {MASTER_ADDR}, Master Port: {MASTER_PORT}, World Size: {WORLD_SIZE}")

    processes = []
    for rank in range(args.nproc_per_node):
        current_env = os.environ.copy()
        current_env["MASTER_ADDR"] = MASTER_ADDR
        current_env["MASTER_PORT"] = MASTER_PORT
        current_env["WORLD_SIZE"] = str(WORLD_SIZE)
        current_env["RANK"] = str(rank)
        current_env["LOCAL_RANK"] = str(rank)  # For single-node, RANK and LOCAL_RANK are the same
        current_env["PYTHONUNBUFFERED"] = "1"

        command = [sys.executable, args.training_script] + args.training_script_args

        print(f"Launching process for RANK {rank} with command: {' '.join(command)}")
        try:
            process = subprocess.Popen(command, env=current_env)
            processes.append(process)
        except Exception as e:
            print(f"Error launching process for RANK {rank}: {e}")
            # If one process fails to launch, terminate already started processes
            for p in processes:
                try:
                    p.terminate()
                    p.wait(timeout=5) # Wait a bit for termination
                except subprocess.TimeoutExpired:
                    p.kill() # Force kill if terminate doesn't work
                except Exception as kill_e:
                    print(f"Error trying to kill process {p.pid}: {kill_e}")
            sys.exit(1)


    print(f"Launched {len(processes)} processes. Waiting for them to complete...")

    # Wait for all processes to complete
    for rank, process in enumerate(processes):
        try:
            process.wait()
            if process.returncode != 0:
                print(f"Process RANK {rank} (PID {process.pid}) exited with error code {process.returncode}.")
        except Exception as e:
            print(f"Error waiting for process RANK {rank} (PID {process.pid}): {e}")

    print("All training processes have completed.")

if __name__ == "__main__":
    main()
