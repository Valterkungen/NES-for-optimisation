# Natural Evolution Strategy (NES) for Optimization

A comprehensive implementation of Natural Evolution Strategy algorithms applied to three challenging optimization problems, developed as part of DSA4212 at Singapore University.

## Project Overview

This project demonstrates the effectiveness of Natural Evolution Strategy (NES) - a gradient-free optimization algorithm that uses natural gradients and the Fisher information matrix for parameter updates. We compare NES performance against traditional gradient descent methods across three distinct optimization domains.

## Key Features

- **Advanced NES Implementation**: Complete Natural Evolution Strategy with adaptive learning rates and natural gradient computation
- **Multi-Domain Applications**: Three real-world optimization problems showcasing algorithm versatility
- **Comparative Analysis**: Systematic comparison between NES and gradient descent methods
- **Comprehensive Benchmarking**: Rosenbrock function validation and performance metrics

## Problems Addressed

### 1. Flame Optimization Problem
**Objective**: Optimize nozzle parameters (center position and width) for maximum flame performance

- **Parameters**: Nozzle center (`nozc`) ∈ [-1.5, 0], Nozzle width (`nozw`) ∈ [0, 4.0]
- **Method**: NES with population-based sampling and natural gradient updates
- **Key Innovation**: Handles complex, non-convex flame dynamics simulation

### 2. Acoustic Control Problem
**Objective**: Minimize acoustic noise through optimal control parameters

- **Parameters**: Control phase (`c_pse`) ∈ [0, 2.0], Control amplitude (`c_amp`) ∈ [-2, 2]
- **Challenge**: Multi-modal optimization landscape with local minima
- **Solution**: NES with fitness shaping and adaptive exploration

### 3. Ice Thickness Estimation Problem
**Objective**: Determine optimal ice thickness parameters using finite element analysis

- **Parameters**: Thickness amplitude (`a`) ∈ [0, 2.0], Displacement (`b`) ∈ [-2, 2]
- **Complexity**: Computationally expensive finite element simulations
- **Approach**: Efficient NES implementation with bounded parameter space

## Technical Implementation

### Natural Evolution Strategy Core
```python
class NaturalEvolutionStrategy:
    def __init__(self, population_size=50, learning_rate=0.1, sigma_init=0.1):
        # Natural gradient computation with Fisher information matrix
        # Adaptive learning rate and exploration control
        # Bounded parameter optimization
```

### Key Algorithms
- **Natural Gradient Computation**: Uses Fisher information matrix for parameter updates
- **Fitness Shaping**: Rank-based utility transformation for robust optimization
- **Adaptive Exploration**: Dynamic σ adjustment based on convergence progress
- **Bounded Optimization**: Constraint handling for real-world parameter limits

## Results Summary

### Performance Comparison (NES vs Gradient Descent)
- **Flame Problem**: NES achieved 15% better performance with more robust convergence
- **Acoustic Problem**: NES successfully avoided local minima, 23% improvement over GD
- **Ice Problem**: Comparable performance with better exploration of parameter space

### Benchmark Results (Rosenbrock Function)
- **Dimensions Tested**: 2D, 5D, 10D, 25D
- **Population Sizes**: 50, 500
- **Convergence Rate**: Consistent sub-linear convergence across all dimensions
- **Scalability**: Maintained performance up to 25 dimensions

## Repository Structure

```
├── src/
│   ├── algorithms/
│   │   └── nes_rosenbrock.py      # Rosenbrock benchmark implementation
│   ├── problems/
│   │   ├── flame_optimization.py   # Flame nozzle optimization
│   │   ├── acoustic_optimization.py # Acoustic control optimization
│   │   └── ice_optimization.py     # Ice thickness estimation
│   └── utils/
│       ├── compute_flame_performance.py # Flame simulation utilities
│       └── TriFEMLibGF24.py       # Finite element library
├── results/
│   └── figures/                   # Key visualization results
├── docs/
│   └── DSA4212_NES.pdf           # Complete technical report
└── README.md
```

## Key Contributions

1. **Robust NES Implementation**: Complete natural evolution strategy with adaptive mechanisms
2. **Multi-Domain Validation**: Demonstrated effectiveness across diverse optimization landscapes
3. **Comparative Analysis**: Systematic evaluation against traditional optimization methods
4. **Real-World Applications**: Practical solutions to engineering optimization problems

## Technical Highlights

- **Natural Gradients**: Leverages geometric structure of parameter space for efficient updates
- **Fisher Information Matrix**: Provides second-order optimization information without Hessian computation
- **Population-Based Sampling**: Robust exploration through stochastic sampling strategies
- **Constraint Handling**: Effective bounded optimization for real-world parameter constraints

## Usage

Each problem can be run independently:

```bash
# Flame optimization
python src/problems/flame_optimization.py

# Acoustic control
python src/problems/acoustic_optimization.py

# Ice thickness estimation
python src/problems/ice_optimization.py

# Rosenbrock benchmark
python src/algorithms/nes_rosenbrock.py
```

## Dependencies

- NumPy: Numerical computations
- JAX: Automatic differentiation and GPU acceleration
- Matplotlib: Visualization and plotting
- SciPy: Scientific computing utilities

## Academic Context

This project was developed for **DSA4212: Optimization for Large-Scale Data-Driven Inference** at Singapore University, demonstrating advanced optimization techniques for complex, real-world problems.

## Results and Visualizations

Key results and performance comparisons are available in the `results/figures/` directory, with comprehensive analysis provided in the technical report (`docs/DSA4212_NES.pdf`).

---

*This implementation showcases the power of natural evolution strategies for challenging optimization problems where traditional gradient-based methods may struggle with local minima, noise, or expensive function evaluations.*
