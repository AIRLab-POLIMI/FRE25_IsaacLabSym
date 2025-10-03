#!/usr/bin/env python3
"""
GPU Memory Capacity Test Script for FRE25 Environment
Tests different environment counts to find optimal GPU usage without OOM
"""

import argparse
import subprocess
import time
import sys

# Test configurations: environment counts to try
TEST_CONFIGS = [
    16,   # Very conservative
    32,   # Conservative
    64,   # Moderate
    128,  # Aggressive
    256,  # Very aggressive
    512,  # Maximum
]

def run_test(num_envs, duration=30):
    """
    Run a quick test with specified number of environments
    Returns True if successful, False if OOM or other error
    """
    print(f"\n{'='*60}")
    print(f"Testing with {num_envs} environments...")
    print(f"{'='*60}")
    
    isaac_lab_path = os.path.expanduser("~/Desktop/PaoloGinefraMultidisciplinaryProject/IsaacLab")
    
    cmd = [
        f"{isaac_lab_path}/isaaclab.sh",
        "-p", "scripts/sb3/train.py",
        "--task", "Fre25-Isaaclabsym-Direct-v0",
        "--num_envs", str(num_envs),
        "--max_iterations", "5",  # Just 5 iterations for testing
        "--headless",
    ]
    
    try:
        # Run for limited time
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Wait for process with timeout
        try:
            stdout, stderr = process.communicate(timeout=duration)
            
            # Check for OOM errors
            if "CUDA out of memory" in stderr or "CUDA out of memory" in stdout:
                print(f"❌ FAILED: CUDA out of memory with {num_envs} envs")
                return False
            
            if "PhysX error" in stderr or "PhysX error" in stdout:
                print(f"❌ FAILED: PhysX error with {num_envs} envs")
                return False
            
            if process.returncode == 0 or "Training started" in stdout:
                print(f"✅ SUCCESS: {num_envs} environments work!")
                return True
            else:
                print(f"⚠️  UNKNOWN: Process ended with code {process.returncode}")
                return False
                
        except subprocess.TimeoutExpired:
            # Timeout is actually good - means it's running
            process.terminate()
            process.wait()
            print(f"✅ SUCCESS: {num_envs} environments running (timed out as expected)")
            return True
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Test GPU capacity for FRE25 environment")
    parser.add_argument("--duration", type=int, default=30, help="Test duration per config (seconds)")
    parser.add_argument("--start-from", type=int, default=None, help="Start from specific env count")
    args = parser.parse_args()
    
    print("="*60)
    print("GPU CAPACITY TEST FOR FRE25")
    print("="*60)
    print("\nThis script will test different environment counts to find")
    print("the maximum your GPU can handle without OOM errors.")
    print(f"\nTest duration per config: {args.duration} seconds")
    print("\nStarting tests...\n")
    
    import os
    
    results = {}
    start_testing = args.start_from is None
    
    for num_envs in TEST_CONFIGS:
        if not start_testing:
            if num_envs >= args.start_from:
                start_testing = True
            else:
                print(f"Skipping {num_envs} environments...")
                continue
        
        success = run_test(num_envs, args.duration)
        results[num_envs] = success
        
        if not success:
            print(f"\n⚠️  Stopping tests - {num_envs} environments failed")
            break
        
        # Wait a bit between tests
        time.sleep(5)
    
    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    max_working = 0
    for num_envs, success in results.items():
        status = "✅ WORKS" if success else "❌ FAILED"
        print(f"{num_envs:>4} environments: {status}")
        if success:
            max_working = num_envs
    
    if max_working > 0:
        print(f"\n🎯 RECOMMENDATION: Use {max_working} environments")
        print(f"   For safety margin, consider using {max_working // 2} environments")
        
        # Create recommended script
        script_path = "RUN_SB3_TRAIN_RECOMMENDED.sh"
        with open(script_path, 'w') as f:
            f.write(f"""#!/bin/bash
# Recommended training script based on GPU capacity test
# Maximum safe environments: {max_working}
# Recommended with safety margin: {max_working // 2}

ISAAC_LAB_PATH="${{HOME}}/Desktop/PaoloGinefraMultidisciplinaryProject/IsaacLab"

echo "Starting training with {max_working // 2} environments (recommended safe value)"

${{ISAAC_LAB_PATH}}/isaaclab.sh -p scripts/sb3/train.py \\
    --task Fre25-Isaaclabsym-Direct-v0 \\
    --num_envs {max_working // 2} \\
    --headless \\
    "$@"
""")
        subprocess.run(["chmod", "+x", script_path])
        print(f"\n📝 Created {script_path} with recommended settings")
    else:
        print("\n❌ No working configuration found!")
        print("   Possible issues:")
        print("   - GPU memory too limited")
        print("   - PhysX settings need adjustment")
        print("   - Other configuration issues")

if __name__ == "__main__":
    main()
