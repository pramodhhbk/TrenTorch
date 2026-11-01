import numpy as np


def mse_gradient(
    input: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray | None,
    target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    input:  shape (batch_size, in_features)
    weight: shape (out_features, in_features)
    bias:   shape (out_features,), or None
    target: shape (batch_size, out_features)

    Returns:
        grad_weight: same shape as weight
        grad_bias: same shape as bias, or None if bias is None
    """
    # TODO: Implement the gradient formulas derived in Theory.
    # Do not use an autograd library -- these are the manual derivatives.
    batch_size,in_features = input.shape
    out_features = weight.shape[0]

    N = batch_size * out_features

    '''if bias is not None: 
        #bias = bias.reshape(out_features,-1)
        y_pred = (input @ weight.T) + bias
        grad_weight = ((2/N) * (y_pred - target)).T @ input
        grad_bias = np.sum((2/N) * (y_pred - target))
        return (grad_weight,grad_bias)

    else:
        y_pred = (input @ weight.T)
        grad_weight = ((2/N) * (y_pred - target)).T @ input
        return (grad_weight,None)'''

    if bias is not None:
        y_pred = (input @ weight.T) + bias
        loss = (1/N) * np.sum(np.power((y_pred - target),2))
        grad_pred = (2/N) * (y_pred - target)
        grad_weight = np.dot(grad_pred.T,input)
        grad_bias = np.sum(grad_pred,axis = 0)
        return (grad_weight,grad_bias)
    else:
        y_pred = (input @ weight.T)
        loss = (1/N) * np.sum(np.power((y_pred - target),2))
        grad_pred = (2/N) * (y_pred - target)
        grad_weight = np.dot(grad_pred.T,input)
        grad_bias = None
        return (grad_weight,grad_bias)
