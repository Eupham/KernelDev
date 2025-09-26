
import subprocess
import sys
import modal

def run_command_in_modal(command):
    print(f"Executing in Modal: {command}")
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            shell=True,
            executable="/bin/bash",
            text=True,
            bufsize=1
        )
        for line in iter(process.stdout.readline, ''):
            print(line, end='')
        process.stdout.close()
        return_code = process.wait()
        if return_code:
            raise subprocess.CalledProcessError(return_code, command)
    except Exception as e:
        print(f"Error running command in modal: {command}\n{e}")
        sys.exit(1)

# Define the Modal App
app = modal.App("kernel-dev-runner")

@app.function(
    gpu="h100",
    secrets=[modal.Secret.from_dict({
        "MODAL_TOKEN_ID": "ak-oqW1DmWZhAEGVkGZSSr5N0",
        "MODAL_TOKEN_SECRET": "as-bQyddw1JYzje8vY4apu8Qi"
    })],
    timeout=3600,
    container_idle_timeout=300
)
def run_training():
    print("--- Setting up remote environment in Modal ---")
    setup_commands = [
        "apt-get update -y",
        "apt-get install -y software-properties-common",
        "add-apt-repository -y ppa:git-core/ppa",
        "apt-get install -y git build-essential",
        "pip install uv",
        "uv pip install --system torch ninja datasets matplotlib triton",
        "rm -rf KernelDev",
        "git clone --branch remove-unnecessary-tasks https://github.com/Eupham/KernelDev.git"
    ]
    for cmd in setup_commands:
        run_command_in_modal(cmd)

    print("\n--- Starting training script in Modal ---")
    # We need to copy the modified config file into the container
    # But since we are running in a different context, let's just create it
    # For simplicity, we assume the launcher has already modified the config.yaml
    # and we just need to run the entry.py with it.
    # A better approach would be to pass the config as an argument or a file.
    # For now, let's assume the git-cloned config is the one to use.

    training_command = (
        "python KernelDev/entry.py "
        "--nproc_per_node=1 "
        "--config KernelDev/config.yaml "
        "--epochs 1 "
        "--precision bf16"
    )
    run_command_in_modal(training_command)

    print("\n--- Training complete ---")
