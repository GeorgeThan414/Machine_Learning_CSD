Machine Learning 
This repository contains each task of the machine learning master course from Aristotle University of Thessaloniki

=====
# Project Description

This repository contains a PyTorch-based implementation for estimating household-level daily per-capita consumption and computing poverty rates using household survey data from the World Bank Poverty Estimation Challenge. The task is formulated as a regression problem at the household level, where predicted consumption values are later aggregated to estimate poverty rates at predefined thresholds.

The proposed approach is based on a deep Multi-Layer Perceptron (MLP) with batch normalization, dropout, and a residual refinement block. The target variable is modeled in logarithmic scale to improve numerical stability and to better handle the heavy-tailed nature of consumption data, especially at low income levels. To improve robustness and generalization across different surveys, multiple models are trained with different random seeds (Mixture of Experts), and the final predictions are calibrated using quantile mapping to better align the predicted and true consumption distributions.

The codebase includes a complete and modular pipeline covering data preprocessing, PyTorch datasets and dataloaders, training and validation loops, official poverty metric computation, and automatic generation of submission files. The overall goal of the project is to provide a clean, reproducible, and extensible solution, with emphasis on metric-aware optimization rather than leaderboard-specific tuning.

# Poverty Prediction Project

Machine Learning project for poverty and household consumption prediction.

## Structure
- model/
- utils/
- submission/
- report.pdf

## How to run
pip install -r requirements.txt



