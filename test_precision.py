#!/usr/bin/env python3
"""
Test script to demonstrate the precision options in the GPT training script.
This script shows how to run training with different precision settings.
"""

import subprocess
import sys
import time

def run_training(precision, epochs=1, batch_size=4):
    """Run training with specified precision."""
    print(f"\n{'='*60}")
    print(f"Testing fp{precision} precision training")
    print(f"{'='*60}")
    
    cmd = [
        sys.executable, "entry.py",
        "--precision", str(precision),
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--seq-len", "512",  # Shorter sequence for faster testing
        "--learning-rate", "1e-4"
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        elapsed_time = time.time() - start_time
        
        print(f"\nTraining completed in {elapsed_time:.2f} seconds")
        print(f"Return code: {result.returncode}")
        
        if result.returncode == 0:
            print("✓ Training successful!")
            # Extract key information from output
            lines = result.stdout.split('\n')
            for line in lines:
                if any(keyword in line.lower() for keyword in ['precision', 'mixed', 'fp16', 'fp32', 'dtype', 'parameters']):
                    print(f"  {line.strip()}")
        else:
            print("✗ Training failed!")
            print("STDERR:")
            print(result.stderr)
            
        return result.returncode == 0, elapsed_time
        
    except subprocess.TimeoutExpired:
        print("✗ Training timed out!")
        return False, 300
    except Exception as e:
        print(f"✗ Error running training: {e}")
        return False, 0

def main():
    """Test both precision modes."""
    print("=== GPT Model Precision Testing ===")
    print("This script tests both fp32 and fp16 precision training modes.")
    
    results = {}
    
    # Test fp32 (baseline)
    success_fp32, time_fp32 = run_training(precision=32, epochs=1, batch_size=4)
    results['fp32'] = {'success': success_fp32, 'time': time_fp32}
    
    # Test fp16 (mixed precision)
    success_fp16, time_fp16 = run_training(precision=16, epochs=1, batch_size=4)
    results['fp16'] = {'success': success_fp16, 'time': time_fp16}
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    for precision, result in results.items():
        status = "✓ SUCCESS" if result['success'] else "✗ FAILED"
        print(f"{precision.upper():>6}: {status:>10} - {result['time']:>6.2f}s")
    
    if results['fp32']['success'] and results['fp16']['success']:
        speedup = results['fp32']['time'] / results['fp16']['time']
        print(f"\nSpeedup with fp16: {speedup:.2f}x")
        
        if speedup > 1.1:
            print("✓ Mixed precision training provides speedup!")
        else:
            print("~ Mixed precision training has similar performance")
    
    print(f"\n{'='*60}")
    print("Usage examples:")
    print("  python entry.py --precision 32                    # fp32 training")
    print("  python entry.py --precision 16                    # fp16/mixed precision")
    print("  python entry.py --precision 16 --batch-size 32    # fp16 with larger batch")
    print("  python entry.py --precision 32 --epochs 5         # fp32 with more epochs")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
