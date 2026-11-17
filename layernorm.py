import numpy as np


EPSILON = 1e-5


class LayerNorm:
    def __init__(self,
                 D_MODEL: int,
                 beta: np.ndarray | None = None,
                 gamma: np.ndarray | None = None):
        self.beta = beta if beta is not None else np.zeros((1, D_MODEL))
        self.gamma = gamma if gamma is not None else np.ones((1, D_MODEL))

    def __call__(self, array: np.ndarray):
        mean = np.sum(array, axis=-1, keepdims=True) / array.shape[0]
        diff = array - mean
        sum_diff = np.sum(np.square(diff), axis=-1, keepdims=True)
        variance = sum_diff / array.shape[0]
        array = diff / (np.sqrt(variance + EPSILON))
        return array * self.gamma + self.beta
