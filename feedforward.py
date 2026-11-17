import numpy as np


class FeedForward:
    def __init__(self, D_MODEL: int,
                 w1: np.ndarray | None = None,
                 w2: np.ndarray | None = None,
                 b1: np.ndarray | None = None,
                 b2: np.ndarray | None = None):
        self.D_MODEL = D_MODEL
        if w1 is not None:
            self.w1 = w1
        else:
            self.w1 = np.random.randn(D_MODEL, D_MODEL*4) * 0.02
        if w2 is not None:
            self.w2 = w2
        else:
            self.w2 = np.random.randn(D_MODEL*4, D_MODEL) * 0.02
        self.b1 = b1 if b1 is not None else np.zeros(D_MODEL*4)
        self.b2 = b2 if b2 is not None else np.zeros(D_MODEL)

    def __call__(self, array):
        x = np.matmul(array, self.w1) + self.b1
        x = np.maximum(x, 0)
        x = np.matmul(x, self.w2) + self.b2
        return x
