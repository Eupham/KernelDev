# Automated Modal Training Launcher

This guide describes how to use the automated launcher script to run the training process on Modal. The script handles all necessary setup, authentication, and execution steps.

## Prerequisites

Before running the launcher, you must perform a one-time setup of the necessary Python packages.

1.  **Install Dependencies:**
    ```bash
    pip install modal pyyaml
    ```

2.  **Set Environment Variables:**
    Ensure you have set the following environment variables with your Modal credentials:
    ```bash
    export MODAL_TOKEN_ID="<your_token_id>"
    export MODAL_TOKEN_SECRET="<your_token_secret>"
    ```

## How to Run

To start the entire process, simply execute the `launcher.py` script. It will:

1.  Read the local `config.yaml`, temporarily modify the number of training samples to 1000 in memory.
2.  Build a custom container image with all necessary system and Python dependencies.
3.  Mount the local project directory into the container.
4.  Launch a job on a Modal H100 GPU instance, passing the modified configuration.
5.  Run the training script (`entry.py`) inside the Modal environment.

Execute the script with the following command:

```bash
python launcher.py
```

The script will provide real-time output from the remote Modal job.