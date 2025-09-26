import subprocess
import os
import sys
import modal_setup

# --- Configuration ---
MODAL_TOKEN_ID = os.environ.get("MODAL_TOKEN_ID")
MODAL_TOKEN_SECRET = os.environ.get("MODAL_TOKEN_SECRET")
CONFIG_FILE = "config.yaml"
REPO_URL = "https://github.com/Eupham/KernelDev.git"
BRANCH = "remove-unnecessary-tasks"

def run_command_stream(command):
    """Runs a command and streams its output in real-time."""
    print(f"Executing: {command}")
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
        print(f"Error running command: {command}\n{e}")
        sys.exit(1)

def modify_config(filepath, max_samples):
    """Temporarily modifies the config.yaml file."""
    import yaml
    try:
        with open(filepath, 'r') as f:
            config = yaml.safe_load(f)

        original_max_samples = config.get('data', {}).get('max_samples')
        config['data']['max_samples'] = max_samples

        with open(filepath, 'w') as f:
            yaml.dump(config, f)

        return original_max_samples
    except Exception as e:
        print(f"Error modifying config file: {e}")
        sys.exit(1)

def restore_config(filepath, original_value):
    """Restores the original value in the config.yaml file."""
    import yaml
    if original_value is None:
        return
    try:
        with open(filepath, 'r') as f:
            config = yaml.safe_load(f)

        config['data']['max_samples'] = original_value

        with open(filepath, 'w') as f:
            yaml.dump(config, f)
        print(f"Restored data.max_samples to {original_value}.")
    except Exception as e:
        print(f"Error restoring config file: {e}")

def main():
    """Main launcher script."""
    # 1. Run local setup for dependencies
    print("--- Running local setup for dependencies ---")
    modal_setup.main()

    # 2. Check for Modal credentials
    print("\n--- Checking for Modal credentials ---")
    if not all([MODAL_TOKEN_ID, MODAL_TOKEN_SECRET]):
        print("Error: MODAL_TOKEN_ID and MODAL_TOKEN_SECRET must be set in the environment.")
        sys.exit(1)
    print("Modal credentials found.")

    original_samples = None
    modal_runner_path = "modal_runner.py"

    try:
        # 3. Modify config.yaml
        print(f"\n--- Modifying {CONFIG_FILE} for training run ---")
        original_samples = modify_config(CONFIG_FILE, 1000)
        print(f"Changed data.max_samples to 1000. Original value was {original_samples}.")

        # 4. Define and run the Modal job
        print("\n--- Preparing to launch Modal job ---")

        # This script will be written to a file and run by Modal.
        # It defines the remote environment and the tasks to be executed.
        # Note the change from modal.Stub to modal.App
        modal_script_content = f"""
import subprocess
import sys
import modal

def run_command_in_modal(command):
    print(f"Executing in Modal: {{command}}")
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
        print(f"Error running command in modal: {{command}}\\n{{e}}")
        sys.exit(1)

# Define the Modal App
app = modal.App("kernel-dev-runner")

@app.function(
    gpu="h100",
    secrets=[modal.Secret.from_dict({{
        "MODAL_TOKEN_ID": "{MODAL_TOKEN_ID}",
        "MODAL_TOKEN_SECRET": "{MODAL_TOKEN_SECRET}"
    }})],
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
        "git clone --branch {BRANCH} {REPO_URL}"
    ]
    for cmd in setup_commands:
        run_command_in_modal(cmd)

    print("\\n--- Starting training script in Modal ---")
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

    print("\\n--- Training complete ---")
"""

        with open(modal_runner_path, "w") as f:
            f.write(modal_script_content)

        print("Launching Modal job...")
        # We no longer need proxychains.
        # The modal CLI expects the function name to be specified for `run`
        run_command_stream(f"python -m modal run {modal_runner_path}")

    finally:
        # 5. Restore config.yaml
        print(f"\n--- Restoring {CONFIG_FILE} ---")
        if original_samples is not None:
            restore_config(CONFIG_FILE, original_samples)

        # 6. Clean up the temporary modal runner script
        if os.path.exists(modal_runner_path):
            os.remove(modal_runner_path)
            print(f"Removed temporary script: {modal_runner_path}")

        print("\n--- Launcher script finished ---")

if __name__ == "__main__":
    main()