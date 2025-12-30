#!/usr/bin/env python3
"""
Test runner script for ML Platform
Provides easy commands to run different types of tests
"""

import sys
import subprocess
import argparse
from pathlib import Path


def run_command(cmd, description):
    """Run a command and print results"""
    print(f"\n{'='*60}")
    print(f"🧪 {description}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
            
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Error running command: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="ML Platform Test Runner")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    parser.add_argument("--api", action="store_true", help="Run API tests only")
    parser.add_argument("--data", action="store_true", help="Run data processing tests only")
    parser.add_argument("--models", action="store_true", help="Run model configuration tests only")
    parser.add_argument("--coverage", action="store_true", help="Run tests with coverage report")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--fast", action="store_true", help="Run only fast tests")
    
    args = parser.parse_args()
    
    # Base pytest command
    base_cmd = "python -m pytest test_ml_platform.py"
    
    if args.verbose:
        base_cmd += " -v"
    
    success = True
    
    if args.all or (not any([args.api, args.data, args.models, args.coverage, args.fast])):
        # Run all tests
        cmd = base_cmd
        if args.coverage:
            cmd += " --cov=backend --cov-report=term-missing --cov-report=html"
        success &= run_command(cmd, "Running All Tests")
        
    else:
        # Run specific test categories
        if args.api:
            cmd = f"{base_cmd} -k TestMLPlatformAPI"
            success &= run_command(cmd, "Running API Tests")
            
        if args.data:
            cmd = f"{base_cmd} -k TestDataProcessing"
            success &= run_command(cmd, "Running Data Processing Tests")
            
        if args.models:
            cmd = f"{base_cmd} -k TestModelConfiguration"
            success &= run_command(cmd, "Running Model Configuration Tests")
            
        if args.fast:
            cmd = f"{base_cmd} -m 'not slow'"
            success &= run_command(cmd, "Running Fast Tests Only")
            
        if args.coverage:
            cmd = f"{base_cmd} --cov=backend --cov-report=term-missing --cov-report=html"
            success &= run_command(cmd, "Running Tests with Coverage")
    
    # Print summary
    print(f"\n{'='*60}")
    if success:
        print("🎉 All tests completed successfully!")
    else:
        print("⚠️  Some tests failed. Check output above.")
    print(f"{'='*60}")
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())