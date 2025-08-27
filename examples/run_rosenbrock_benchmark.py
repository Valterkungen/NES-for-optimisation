#!/usr/bin/env python3
"""
Example script demonstrating Rosenbrock function optimization using Natural Evolution Strategy.

This script benchmarks the NES algorithm on the classic Rosenbrock function
across multiple dimensions and population sizes.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.algorithms.nes_rosenbrock import produce_results

if __name__ == "__main__":
    print("Running Rosenbrock Function Benchmark with Natural Evolution Strategy")
    print("=" * 70)
    
    # Run the benchmark
    produce_results()
    
    print("\nRosenbrock benchmark completed successfully!")
    print("Results have been saved to JSON file and printed above.")
