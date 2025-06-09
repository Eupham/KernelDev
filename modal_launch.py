#!/usr/bin/env python3
"""
Modal launcher for KernelDev training with GPU acceleration.
Can be run either with Modal or as a standalone Python script for local testing.
"""

import os
import sys
import time

# Check if we're running with Modal
using_modal = 'modal' in sys.argv[0].lower()

# Only import modal when running with Modal
if using_modal:
    import modal
    
    # Create Modal app and volume
    app = modal.App("kerneldev-training")
    vol = modal.Volume.from_name("kerneldev-vol", create_if_missing=True)
    
    # Create base image with CUDA support and required packages
    image = modal.Image.debian_slim(python_version="3.10").apt_install(
        "build-essential", 
        "ninja-build",
        "git"
    ).pip_install([
        "torch>=2.0.0",
        "torchvision>=0.15.0", 
        "triton>=2.0.0",
        "matplotlib>=3.0.0",
        "datasets>=2.0.0",
        "pyyaml>=6.0.0",
        "tqdm>=4.65.0",
        "numpy>=1.21.0",
        "huggingface-hub>=0.16.0",
        "transformers>=4.21.0",
        "tokenizers>=0.13.0"
    ])
    
    # Add all local Python modules to the image
    image = image.add_local_dir(".", "/root/kerneldev", copy=True)
    
    # Set working directory and install project
    image = image.run_commands([
        "cd /root/kerneldev",
        "ls -la",  # Debug: show what files are copied
        "python -c 'import torch; print(f\"PyTorch version: {torch.__version__}\"); print(f\"CUDA available: {torch.cuda.is_available()}\")'",
    ])


def run_training_locally():
    """
    Run training directly without Modal for local testing.
    """
    print("Running training locally...")
    
    # Import and run the main entry point
    try:
        # Add current directory to path if needed
        if '.' not in sys.path:
            sys.path.insert(0, '.')
        
        from entry import main as entry_main
        
        # Run with default configuration
        print("Starting local training...")
        entry_main()
        
    except ImportError as e:
        print(f"Error importing entry module: {e}")
        print("Make sure you're running this from the KernelDev directory")
        return "Import error"
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return f"Training error: {e}"
    
    return "Local training complete!"


def run_inference_locally():
    """
    Run inference testing locally.
    """
    print("Running inference testing locally...")
    
    try:
        # Add current directory to path if needed
        if '.' not in sys.path:
            sys.path.insert(0, '.')
        
        # Import test modules
        from test_inference import main as test_inference_main
        
        print("Starting inference tests...")
        test_inference_main()
        
    except ImportError as e:
        print(f"Error importing test modules: {e}")
        return "Import error"
    except Exception as e:
        print(f"Inference testing failed: {e}")
        import traceback
        traceback.print_exc()
        return f"Inference error: {e}"
    
    return "Local inference testing complete!"


# Define Modal-specific functions if we're using Modal
if using_modal:
    @app.function(
        gpu="H100", 
        volumes={"/data": vol}, 
        timeout=7200,  # 2 hours
        image=image,
        memory=32768,  # 32GB RAM
    )
    def train_h100():
        """
        Main training function for Modal using H100.
        """
        print("Running training on H100...")
        print("Working directory:", os.getcwd())
        print("Files in current directory:", os.listdir("."))
        
        # Change to project directory
        os.chdir("/root/kerneldev")
        print("Changed to kerneldev directory")
        print("Files in kerneldev directory:", os.listdir("."))
        
        # Set up environment
        os.environ['PYTHONPATH'] = '/root/kerneldev'
        
        # Import and run training
        try:
            import sys
            sys.path.insert(0, '/root/kerneldev')
            
            # Run the entry point
            import subprocess
            result = subprocess.run([
                sys.executable, "entry.py", 
                "--config", "config.yaml",
                "--precision", "16"  # Use mixed precision on H100
            ], capture_output=True, text=True, cwd="/root/kerneldev")
            
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            print("Return code:", result.returncode)
            
            # Save any generated plots to volume
            try:
                import shutil
                os.makedirs('/data/results', exist_ok=True)
                
                # Copy checkpoints and plots if they exist
                if os.path.exists('checkpoints'):
                    shutil.copytree('checkpoints', '/data/results/checkpoints', dirs_exist_ok=True)
                    print("Copied checkpoints to volume")
                
                # Copy any PNG files (plots)
                for file in os.listdir('.'):
                    if file.endswith('.png'):
                        shutil.copy(file, f'/data/results/{file}')
                        print(f"Copied {file} to volume")
                        
            except Exception as e:
                print(f"Error copying results to volume: {e}")
            
            return f"Training completed with return code: {result.returncode}"
            
        except Exception as e:
            print(f"Training failed: {e}")
            import traceback
            traceback.print_exc()
            return f"Training error: {e}"
    
    @app.function(
        gpu="A100", 
        volumes={"/data": vol}, 
        timeout=7200,  # 2 hours
        image=image,
        memory=24576,  # 24GB RAM
    )
    def train_a100():
        """
        Training function for Modal using A100 for comparison.
        """
        print("Running training on A100...")
        
        # Change to project directory
        os.chdir("/root/kerneldev")
        
        # Set up environment
        os.environ['PYTHONPATH'] = '/root/kerneldev'
        
        try:
            import sys
            sys.path.insert(0, '/root/kerneldev')
            
            # Run the entry point with A100-optimized settings
            import subprocess
            result = subprocess.run([
                sys.executable, "entry.py", 
                "--config", "config.yaml",
                "--precision", "16",  # Use mixed precision
                "--batch_size", "8"   # Smaller batch size for A100
            ], capture_output=True, text=True, cwd="/root/kerneldev")
            
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            
            # Save results
            try:
                import shutil
                os.makedirs('/data/results_a100', exist_ok=True)
                
                if os.path.exists('checkpoints'):
                    shutil.copytree('checkpoints', '/data/results_a100/checkpoints', dirs_exist_ok=True)
                
                for file in os.listdir('.'):
                    if file.endswith('.png'):
                        shutil.copy(file, f'/data/results_a100/{file}')
                        
            except Exception as e:
                print(f"Error copying results: {e}")
            
            return f"A100 training completed with return code: {result.returncode}"
            
        except Exception as e:
            return f"A100 training error: {e}"
    
    @app.function(
        gpu="H100", 
        volumes={"/data": vol}, 
        timeout=3600,  # 1 hour
        image=image,
        memory=16384,  # 16GB RAM
    )
    def test_inference():
        """
        Inference testing function for Modal.
        """
        print("Running inference testing on H100...")
        
        # Change to project directory
        os.chdir("/root/kerneldev")
        
        # Set up environment
        os.environ['PYTHONPATH'] = '/root/kerneldev'
        
        try:
            import sys
            sys.path.insert(0, '/root/kerneldev')
            
            # Run inference tests
            import subprocess
            result = subprocess.run([
                sys.executable, "test_inference.py"
            ], capture_output=True, text=True, cwd="/root/kerneldev")
            
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            
            return f"Inference testing completed with return code: {result.returncode}"
            
        except Exception as e:
            return f"Inference testing error: {e}"
    
    @app.function(
        gpu="H100", 
        volumes={"/data": vol}, 
        timeout=3600,  # 1 hour
        image=image,
        memory=16384,  # 16GB RAM
    )
    def test_precision():
        """
        Precision testing function for Modal.
        """
        print("Running precision testing on H100...")
        
        # Change to project directory
        os.chdir("/root/kerneldev")
        
        # Set up environment
        os.environ['PYTHONPATH'] = '/root/kerneldev'
        
        try:
            import sys
            sys.path.insert(0, '/root/kerneldev')
            
            # Run precision tests
            import subprocess
            result = subprocess.run([
                sys.executable, "test_precision.py"
            ], capture_output=True, text=True, cwd="/root/kerneldev")
            
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            
            return f"Precision testing completed with return code: {result.returncode}"
            
        except Exception as e:
            return f"Precision testing error: {e}"
    
    @app.local_entrypoint()
    def main():
        """
        Local entrypoint for Modal deployment.
        """
        if len(sys.argv) > 1:
            command = sys.argv[1].lower()
            
            if command == "a100":
                print("Launching training on A100...")
                result = train_a100.remote()
                print(f"A100 training result: {result}")
                
            elif command == "inference":
                print("Launching inference testing...")
                result = test_inference.remote()
                print(f"Inference testing result: {result}")
                
            elif command == "precision":
                print("Launching precision testing...")
                result = test_precision.remote()
                print(f"Precision testing result: {result}")
                
            elif command == "train" or command == "h100":
                print("Launching training on H100...")
                result = train_h100.remote()
                print(f"H100 training result: {result}")
                
            else:
                print(f"Unknown command: {command}")
                print("Available commands: train, h100, a100, inference, precision")
                
        else:
            # Default: run training on H100
            print("Launching default training on H100...")
            result = train_h100.remote()
            print(f"Training result: {result}")

else:
    # Python-only execution path
    def main():
        """
        Main entry point when running with Python directly (local testing).
        """
        if len(sys.argv) > 1:
            command = sys.argv[1].lower()
            
            if command == "inference":
                return run_inference_locally()
            elif command == "train":
                return run_training_locally()
            else:
                print(f"Unknown local command: {command}")
                print("Available local commands: train, inference")
                return "Invalid command"
        else:
            # Default: run training locally
            return run_training_locally()


if __name__ == "__main__":
    if not using_modal:
        result = main()
        print(f"Local execution result: {result}")
    else:
        main()
