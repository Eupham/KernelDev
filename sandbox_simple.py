#!/usr/bin/env python3
"""
Simple Modal Sandbox for Quick KernelDev Testing

This is a minimal version of the Modal sandbox script that focuses on
running specific tests or commands rather than a comprehensive setup.

Usage:
    python sandbox_simple.py                    # Run attention tests
    python sandbox_simple.py --command "python entry.py --help"
    python sandbox_simple.py --gpu H100         # Use H100 instead of A100
"""

import modal
import argparse
from modal.stream_type import StreamType

def create_parser():
    parser = argparse.ArgumentParser(description="Simple Modal sandbox for KernelDev")
    parser.add_argument("--command", "-c", default="python test_attention_behaviors.py",
                        help="Command to run in the sandbox")
    parser.add_argument("--gpu", default="A100", 
                        choices=["A100", "H100", "T4"],
                        help="GPU type to use")
    parser.add_argument("--timeout", type=int, default=3600,
                        help="Sandbox timeout in seconds (default: 1 hour)")
    return parser

def main():
    args = create_parser().parse_args()
    
    APP_NAME = "kerneldev-sandbox-simple"
    VOLUME_NAME = "kerneldev-cache"

    app = modal.App.lookup(APP_NAME, create_if_missing=True)
    vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

    # Create lightweight image
    image = (
        modal.Image.debian_slim(python_version="3.11")
        .apt_install("git", "build-essential")
        .run_commands(
            "pip install --upgrade pip",
            "pip install --index-url https://download.pytorch.org/whl/cu121 torch",
            "pip install triton datasets matplotlib pyyaml",
        )
        .workdir("/workspace")
    )

    print(f"Creating sandbox with {args.gpu} GPU...")
    print(f"Will run command: {args.command}")
    
    with modal.enable_output():
        sb = modal.Sandbox.create(
            app=app,
            image=image,
            workdir="/workspace",
            volumes={"/data": vol},
            gpu=args.gpu,
            timeout=args.timeout,
            verbose=True,
        )

    # Clone repository if not exists
    sb.exec(
        "bash", "-c",
        "cd /workspace && "
        "test -d KernelDev || git clone --branch remove-unnecessary-tasks "
        "https://github.com/Eupham/KernelDev.git"
    ).wait()

    # Quick environment check
    sb.exec(
        "bash", "-c",
        "cd /workspace/KernelDev && "
        "python -c 'import torch; print(f\"GPU: {torch.cuda.get_device_name() if torch.cuda.is_available() else \"None\"}\")'"
    ).wait()

    # Run the specified command
    print(f"\n=== Running: {args.command} ===")
    result = sb.exec("bash", "-c", f"cd /workspace/KernelDev && {args.command}")
    
    try:
        result.wait()
        print("✅ Command completed successfully")
    except Exception as e:
        print(f"⚠️  Command encountered issues: {e}")

    print(f"\n🚀 Sandbox ID: {sb.object_id}")
    print("Use modal app logs to see detailed output")

if __name__ == "__main__":
    main()