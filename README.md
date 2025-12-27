# 📈 Linear Regression with PyTorch

## Introduction
This project demonstrates a **simple linear regression model implemented using PyTorch**.  
The model learns a linear relationship from synthetic data using gradient descent and visualizes both predictions and loss curves.

The true function used to generate the data is:

y = 0.7x + 0.3

---

## Table of Contents
- Project Overview
- Features
- Installation
- Usage
- Model Architecture
- Training Process
- Visualization
- Dependencies
- Troubleshooting
- License

---

## Project Overview
The script performs the following steps:
1. Generates synthetic linear data
2. Splits data into training and testing sets
3. Defines a custom PyTorch linear regression model
4. Trains the model using L1 loss
5. Evaluates performance on test data
6. Visualizes predictions and loss curves

---

## Features
- Custom PyTorch `nn.Module`
- Manual training loop
- Train/test data split
- Prediction visualization
- Training and test loss curves
- Efficient inference using `torch.inference_mode()`

---

## Installation

### Prerequisites
- Python 3.8+

### Install dependencies
pip install torch matplotlib numpy

### Usage
Run the script using:
python linear_regression.py

During execution:
Training loss is printed each epoch
Model parameters are logged every 10 epochs
Prediction results are plotted
Training and test loss curves are displayed

### Model Architecture
class LinearRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(1))
        self.bias = nn.Parameter(torch.randn(1))

    def forward(self, x):
        return self.weights * x + self.bias
        
The model learns one weight and one bias
Implements a basic linear equation

### Training Process
Loss Function: L1 Loss (nn.L1Loss)
Optimizer: Stochastic Gradient Descent (SGD)
Learning Rate: 0.01
Epochs: 200

Training loop steps:
Forward pass
Loss calculation
Backpropagation
Weight update
Evaluation on test data

### Visualization

Training data (blue)
Test data (green)
Predictions (red)

Loss curves for training and testing

These plots help verify convergence and generalization.

<img width="1245" height="957" alt="image" src="https://github.com/user-attachments/assets/75f1d342-e2b2-47df-a8f3-f9a28468f54d" />
<img width="802" height="681" alt="image" src="https://github.com/user-attachments/assets/c6731a52-4026-46ee-a35b-fbcfb140b612" />

### Dependencies

torch
matplotlib
numpy

### Troubleshooting

Loss not decreasing : Reduce learning rate 
                      Increase number of epochs

Incorrect predictions : Ensure model is in evaluation mode during inference
                        Check tensor shapes (input must be 2D)

### License
MIT License
Free to use, modify, and distribute.



