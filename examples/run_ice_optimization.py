#!/usr/bin/env python3
"""
Example script demonstrating ice thickness optimization using Natural Evolution Strategy.

This script optimizes ice thickness parameters using finite element analysis
and compares NES performance with gradient descent.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.problems.ice_optimization import run_comparison

if __name__ == "__main__":
    print("Running Ice Thickness Optimization Comparison")
    print("=" * 60)
    
    # Run the comparison between NES and Gradient Descent
    results = run_comparison(num_runs=3)
    
    print("\nIce optimization comparison completed successfully!")
    print("Results have been printed above.")
