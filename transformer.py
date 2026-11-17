import numpy as np
import layernorm
import attention
import feedforward


class Transformer:
    def __init__(self, BETA, GAMMA, D_MODEL, D_HEAD,
                 wq: np.ndarray | None = None,
                 wk: np.ndarray | None = None,
                 wv: np.ndarray | None = None,
                 w1: np.ndarray | None = None,
                 w2: np.ndarray | None = None,
                 b1: np.ndarray | None = None,
                 b2: np.ndarray | None = None,
                 beta1: np.ndarray | None = None,
                 gamma1: np.ndarray | None = None,
                 beta2: np.ndarray | None = None,
                 gamma2: np.ndarray | None = None):
        self.BETA = BETA
        self.GAMMA = GAMMA
        self.D_MODEL = D_MODEL
        self.D_HEAD = D_HEAD
        self.attention = attention.Attention(self.D_MODEL, self.D_HEAD,
                                             wq, wk, wv)
        self.ffn = feedforward.FeedForward(self.D_MODEL,
                                           w1, w2, b1, b2)
        self.layernorm1 = layernorm.LayerNorm(D_MODEL, beta1, gamma1)
        self.layernorm2 = layernorm.LayerNorm(D_MODEL, beta2, gamma2)

    def __call__(self, x):
        y = np.array(self.layernorm1(x))
        # print(y.shape)
        x += self.attention(y)
        # print(x.shape)
        y = np.array(self.layernorm2(x))
        # print(y.shape)
        x += self.ffn(y)
        # print(x.shape)
        return x
