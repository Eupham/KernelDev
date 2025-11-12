import modal
import os
import sys
import yaml
import subprocess
from pathlib import Path

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
        "uv pip install --system --index-url https://pypi.org/simple 'torch>=2.0' ninja datasets matplotlib triton pyyaml safensors 'pyarrow==14.0.1' 'numpy<2'",
    )
    .add_local_dir(".", remote_path="/root/KernelDev")
)

# Define the Modal App
app = modal.App(
    "kernel-dev-runner",
    image=image,
    secrets=[modal.Secret.from_local_environ(["MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET"])],
)

volume = modal.Volume.from_name("checkpoint-volume", create_if_missing=True)

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

    # Add the project directory to the Python path so we can import modules
    sys.path.append("/root/KernelDev")

    import json
    from train_loop import Trainer, TrainingConfig

    # Initialize a temporary trainer to find the latest checkpoint
    temp_config = TrainingConfig(checkpoint_dir=config['training']['checkpoint_dir'])
    temp_trainer = Trainer(model=None, config=temp_config, data_builder=None)
    latest_checkpoint = temp_trainer.find_latest_checkpoint()

    if latest_checkpoint:
        print(f"Found checkpoint, loading config from: {latest_checkpoint}")
        try:
            with open(Path(latest_checkpoint) / 'metadata.json', 'r') as f:
                metadata = json.load(f)
            # Override the model config with the one from the checkpoint
            config['model'] = metadata['config']['model']
            # The vocab_size is not in the model dict, so manually set it.
            # In a more robust system, this would also be in the checkpoint.
            config['model']['vocab_size'] = 50257
            print("Overrode current model config with checkpoint config.")
        except (FileNotFoundError, KeyError) as e:
            print(f"Could not load metadata from checkpoint, running with base config. Error: {e}")

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
    config['data']['max_samples'] = 5000000
    print(f"Setting max_samples to {config['data']['max_samples']} for this run.")

    # Define volume name and paths consistently
    volume_name = "checkpoint-volume"
    data_dir = "/data"
    config['training']['checkpoint_dir'] = str(Path(data_dir) / "checkpoints")
    config['data']['dataset_cache_dir'] = str(Path(data_dir) / "dataset")


    # 3. Run the remote function and wait for it to complete.
    print("\n" + "="*50)
    print(">>> LAUNCHING TRAINING RUN")
    print("="*50 + "\n")

    run_training.remote(config)

    print("--- Modal training run finished ---")
