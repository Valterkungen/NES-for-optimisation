from setuptools import setup, find_packages

setup(
    name="nes-optimization",
    version="1.0.0",
    description="Natural Evolution Strategy implementation for multi-domain optimization",
    author="Albin Liljefors",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "jax>=0.3.0",
        "jaxlib>=0.3.0",
        "matplotlib>=3.5.0",
        "scipy>=1.7.0",
        "pandas>=1.3.0",
    ],
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
