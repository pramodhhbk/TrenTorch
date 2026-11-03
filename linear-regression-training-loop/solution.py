import numpy as np


def train_linear_regression(
    input: np.ndarray,
    target: np.ndarray,
    lr: float,
    epochs: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    input:  shape (batch_size, in_features)
    target: shape (batch_size,), one target value per sample
    lr: learning rate
    epochs: number of full-batch gradient-descent steps

    Returns:
        weight: shape (1, in_features)
        bias: shape (1,)
    """
    # TODO: Initialize weight to zeros (1, in_features), bias to zeros (1,).
    # Reshape target to (batch_size, 1) once, up front.
    # Repeat for `epochs` iterations:
    #   1. Compute gradients with mse_gradient()
    #   2. Update weight, bias with gd_step()
    # Reuse those functions -- don't reimplement their logic here.
    batch_size = input.shape[0]
    in_features = input.shape[1]
    weight = np.zeros((1,in_features))
    bias = np.zeros((1,))
    target = target.reshape(batch_size,1)
    for i in range(epochs):
        grad_tuple = mse_gradient(input,weight,bias,target)
        grad_weight = grad_tuple[0]
        grad_bias = grad_tuple[1]
        updated_val = gd_step(weight,bias,grad_weight,grad_bias,lr)
        weight = updated_val[0]
        bias = updated_val[1]

    return (weight,bias)
