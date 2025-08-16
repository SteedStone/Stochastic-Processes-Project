# LINMA1731 - Stochastic Processes Project: Lorenz Particle Tracking

This project implements and analyzes particle filtering techniques for tracking particles following the Lorenz attractor dynamics. The work was completed as part of the LINMA1731 Stochastic Processes course.

## Project Overview

The project consists of two main parts:

### 1. Theoretical Foundation and Lorenz System Analysis
- Implementation of the Lorenz dynamical system with parameters σ=10, ρ=28, β=8/3
- Analysis of empirical probability density functions using box-counting methods
- Visualization of Lorenz attractor projections and 3D trajectories
- Parameter sensitivity analysis using distance metrics (KL divergence, Bhattacharyya distance)

### 2. Particle Filter Implementation (SIR Algorithm)
- Implementation of Sequential Importance Resampling (SIR) particle filter
- Comparison of three resampling methods:
  - Multinomial resampling
  - Residual resampling  
  - Systematic resampling
- Performance analysis with varying:
  - Process noise levels
  - Time step sizes
  - Number of particles
  - Observation noise levels

## Key Files

- `projectP1.ipynb` - Main analysis notebook with Lorenz system implementation and probability density analysis
- `projectP2.ipynb` - Particle filter implementation and performance comparisons
- `Project_student.py` - Python implementation file
- `kalman-filter.py` - Additional filtering implementations
- `images/` - Generated plots and visualizations
- `repport.pdf` - **[Final project report with detailed results and analysis](./repport.pdf)**

## Results Summary

The project demonstrates:
- Effective tracking of chaotic Lorenz trajectories using particle filters
- Performance comparison showing systematic resampling generally outperforms other methods
- Analysis of how process noise, observation noise, and particle count affect filtering accuracy
- Visualization of particle filter performance under various parameter configurations

## Technical Implementation

- **Language**: Python
- **Key Libraries**: NumPy, SciPy, Matplotlib, ipywidgets
- **Integration Method**: Runge-Kutta 4th order (RK4)
- **Filtering**: Sequential Importance Resampling (SIR)

## Results and Analysis

For detailed results, performance metrics, and comprehensive analysis of the particle filtering techniques, please refer to the **[complete project report](./repport.pdf)**.

The report includes:
- Theoretical background on stochastic filtering
- Detailed performance comparisons between resampling methods
- Statistical analysis with RMSE measurements
- Visualization of filtering accuracy under different conditions
- Conclusions and recommendations for optimal parameter selection