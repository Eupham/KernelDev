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

volume = modal.Volume.from_name("kernel-dev-storage", create_if_missing=True)

@app.function(
    gpu="h100",
    timeout=86400,
    scaledown_window=300,
    volumes={"/data": volume},
)
def run_training(config: dict):
    """
    This function runs the training script in a remote Modal container.
    """
    volume.reload()
    print(f"--- Starting training run with optimized kernel ---")

    # The config is passed in as an argument, so we write it to a file
    # in the container for the training script to use.
    remote_config_path = "/root/KernelDev/remote_config.yaml"
    with open(remote_config_path, 'w') as f:
        yaml.dump(config, f)

    training_command = (
        f"python -u /root/KernelDev/entry.py "
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

    volume.commit()

    if return_code != 0:
        print(f"Training script exited with non-zero code: {return_code}")
        sys.exit(return_code)

    print(f"--- Training with optimized kernel complete ---")


@app.local_entrypoint()
def main():
    """
    This is the local entrypoint for the launcher script.
    It runs the training with the optimized kernel.
    """
    print("--- Preparing and launching Modal training job ---")

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
    config['data']['max_samples'] = 500000
    print(f"Setting max_samples to {config['data']['max_samples']} for this run.")

    # 3. Run the remote function and wait for it to complete.
    print("\n" + "="*50)
    print(">>> LAUNCHING TRAINING RUN")
    print("="*50 + "\n")

    run_training.remote(config)

    print("--- Modal training run finished ---")
