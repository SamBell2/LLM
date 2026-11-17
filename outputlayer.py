import numpy as np


class OutputLayer:
    def __init__(self, D_MODEL: int, VOCABULARY_SIZE: int,
                 matrix: np.ndarray | None = None):
        self.D_MODEL = D_MODEL
        self.VOCABULARY_SIZE = VOCABULARY_SIZE
        if matrix is None:
            self.matrix = np.random.randn(D_MODEL, VOCABULARY_SIZE) * 0.02
        else:
            self.matrix = matrix

    def __call__(self, x):
        return np.matmul(x, self.matrix)
