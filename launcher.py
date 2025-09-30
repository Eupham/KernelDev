import modal
import os
import sys
import yaml
import subprocess

# --- Configuration ---
CONFIG_FILE = "config.yaml"

# Define a Modal Image. This is the modern way to define the remote environment.
# It includes all necessary system packages and Python libraries.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .run_commands(
        "apt-get update -y",
        "apt-get install -y software-properties-common build-essential",
        "pip install uv",
        "uv pip install --system torch ninja datasets matplotlib triton pyyaml",
    )
    .add_local_dir(".", remote_path="/root/KernelDev")
)

# Define the Modal App
app = modal.App(
    "kernel-dev-runner",
    image=image,
    secrets=[modal.Secret.from_local_environ(["MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET"])],
)

@app.function(gpu="h100", timeout=3600, scaledown_window=300)
def run_training(config: dict, use_optimized_kernel: bool = False):
    """
    This function runs the training script in a remote Modal container.
    """
    kernel_type = "OPTIMIZED" if use_optimized_kernel else "ORIGINAL"
    print(f"--- Starting training run with {kernel_type} kernel ---")

    # The config is passed in as an argument, so we write it to a file
    # in the container for the training script to use.
    remote_config_path = "/root/KernelDev/remote_config.yaml"
    with open(remote_config_path, 'w') as f:
        yaml.dump(config, f)

    # Set environment variable for kernel selection
    env_vars = f"export USE_OPTIMIZED_KERNEL={'1' if use_optimized_kernel else '0'}; "

    training_command = (
        f"{env_vars}"
        f"python /root/KernelDev/entry.py "
        f"--nproc_per_node=1 "
        f"--config {remote_config_path} "
        f"--epochs 1 "
        f"--precision bf16"
    )

    # Use subprocess to run the command and stream output
    process = subprocess.Popen(
        training_command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        shell=True,
        executable="/bin/bash",
        text=True,
        bufsize=1,
    )

    for line in iter(process.stdout.readline, ""):
        print(line, end="")

    return_code = process.wait()
    if return_code != 0:
        print(f"Training script exited with non-zero code: {return_code}")
        # We don't exit here to allow the next benchmark run to proceed
        # sys.exit(return_code)

    print(f"--- Training with {kernel_type} kernel complete ---")


@app.local_entrypoint()
def main():
    """
    This is the local entrypoint for the launcher script.
    It runs a comparative benchmark between the original and optimized kernels.
    """
    print("--- Preparing and launching Modal benchmark job ---")

    # 1. Check for Modal credentials
    if not all([os.environ.get("MODAL_TOKEN_ID"), os.environ.get("MODAL_TOKEN_SECRET")]):
        print("Error: MODAL_TOKEN_ID and MODAL_TOKEN_SECRET must be set in the environment.")
        sys.exit(1)
    print("Modal credentials found.")

    # 2. Read and modify the configuration in memory
    try:
        with open(CONFIG_FILE, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: {CONFIG_FILE} not found.")
        sys.exit(1)

    print(f"Original max_samples: {config.get('data', {}).get('max_samples')}")
    config['data']['max_samples'] = 1000
    print(f"Temporarily setting max_samples to {config['data']['max_samples']} for this run.")

    # 3. Run the remote function for both kernels and wait for them to complete.
    print("\n" + "="*50)
    print(">>> LAUNCHING BENCHMARK RUNS")
    print("="*50 + "\n")

    # Use starmap to run both function calls and wait for them to finish.
    # The list of tuples corresponds to the arguments of run_training (config, use_optimized_kernel)
    for _ in run_training.starmap([
        (config, False), # Arguments for the first call
        (config, True)   # Arguments for the second call
    ]):
        # This loop will iterate as each job completes, effectively waiting for both.
        pass

    print("--- Modal benchmark finished ---")