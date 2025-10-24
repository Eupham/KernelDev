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
        "uv pip install --system torch ninja datasets matplotlib triton pyyaml typer",
    )
    .add_local_dir(".", remote_path="/root/KernelDev")
)

# Define the Modal App
volume = modal.Volume.from_name("kernel-dev-volume", create_if_missing=True)
app = modal.App(
    "kernel-dev-runner",
    image=image,
    secrets=[modal.Secret.from_local_environ(["MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET"])],
)

@app.function(gpu="h100", timeout=28800, scaledown_window=300, volumes={"/root/data": volume})
def run_training(config: dict):
    """
    This function runs the training script in a remote Modal container.
    """
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

    try:
        for line in iter(process.stdout.readline, ""):
            print(line, end="")

        return_code = process.wait()
        if return_code != 0:
            print(f"Training script exited with non-zero code: {return_code}")
            sys.exit(return_code)

        print(f"--- Training with optimized kernel complete ---")
    finally:
        volume.commit()


def run_training_remotely(config_overrides=None):
    """
    This function runs the training with the optimized kernel.
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

    if config_overrides:
        config.update(config_overrides)

    print(f"Original max_samples: {config.get('data', {}).get('max_samples')}")
    config['data']['max_samples'] = 500000
    print(f"Setting max_samples to {config['data']['max_samples']} for this run.")

    # Add volume path to config
    config['volume_path'] = "/root/data"

    # 3. Run the remote function and wait for it to complete.
    print("\n" + "="*50)
    print(">>> LAUNCHING TRAINING RUN")
    print("="*50 + "\n")

    run_training.remote(config)

    print("--- Modal training run finished ---")

@app.local_entrypoint()
def main(resume: bool = False, clear_volume: bool = False):
    """
    Train the model.
    """
    if clear_volume:
        print("Clearing volume...")
        try:
            modal.Volume.delete("kernel-dev-volume")
            print("Volume 'kernel-dev-volume' deleted.")
        except Exception as e:
            print(f"Could not delete volume: {e}")

        # Re-create the volume
        global volume
        volume = modal.Volume.from_name("kernel-dev-volume", create_if_missing=True)
        print("Volume 'kernel-dev-volume' re-created.")

    if resume:
        print("Resuming training...")
        config_overrides = {'training': {'auto_resume': True}}
        run_training_remotely(config_overrides=config_overrides)
    else:
        print("Starting a new training run...")
        config_overrides = {'training': {'auto_resume': False}}
        run_training_remotely(config_overrides=config_overrides)
