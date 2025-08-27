#!/usr/bin/env python3
"""
Example script demonstrating acoustic control optimization using Natural Evolution Strategy.

This script optimizes control parameters to minimize acoustic noise using NES
compared to gradient descent methods.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.problems.acoustic_optimization import run_comparison, plot_comparison

if __name__ == "__main__":
    print("Running Acoustic Control Optimization Comparison")
    print("=" * 60)
    
    # Run the comparison between NES and Gradient Descent
    results = run_comparison(num_runs=3)
    
    # Generate comparison plots
    plot_comparison(results)
    
    print("\nOptimization comparison completed successfully!")
    print("Check the generated plots for detailed results.")
