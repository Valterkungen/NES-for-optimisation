#!/usr/bin/env python3
"""
Example script demonstrating flame optimization using Natural Evolution Strategy.

This script optimizes nozzle parameters (center position and width) to maximize
flame performance using the NES algorithm.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.problems.flame_optimization import run_optimization

if __name__ == "__main__":
    print("Running Flame Optimization with Natural Evolution Strategy")
    print("=" * 60)
    
    # Run the optimization
    optimizer = run_optimization()
    
    print("\nOptimization completed successfully!")
    print("Check the generated plots for detailed results.")
