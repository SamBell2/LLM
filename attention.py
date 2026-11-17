import numpy as np

import softmax


class Attention:
    def __init__(self, D_MODEL: int, D_HEAD: int,
                 wq: np.ndarray | None = None,
                 wk: np.ndarray | None = None,
                 wv: np.ndarray | None = None):
        if wq is not None:
            self._wq = wq
        else:
            self._wq = np.random.randn(D_MODEL, D_HEAD) * 0.02
        if wk is not None:
            self._wk = wk
        else:
            self._wk = np.random.randn(D_MODEL, D_HEAD) * 0.02
        if wv is not None:
            self._wv = wv
        else:
            self._wv = np.random.randn(D_MODEL, D_HEAD) * 0.02
        self.D_MODEL = D_MODEL

    def __call__(self, embeddings: np.ndarray) -> np.ndarray:
        key_embeddings = np.matmul(embeddings, self._wk)
        query_embeddings = np.matmul(embeddings, self._wq)
        value_embeddings = np.matmul(embeddings, self._wv)
        # grid = []
        # for b in range(embeddings.shape[0]):
        #     grid.append([])
        #     for i in range(len(key_embeddings[0])):
        #         grid[-1].append([])
        #         for j in range(len(key_embeddings[0])):
        #             # i and j swapped in indices to make array look more
        #             # like greencode's anumation
        #             grid[-1][-1].append(np.dot(
        #                 key_embeddings[b][j],
        #                 query_embeddings[b][i]) / np.sqrt(self._wk.shape[-1]))
        # # Example
        # #    Key 0   Key 1   Key 2
        # # Q0  0.5    0.6     0.15
        # # Q1  num    num     num
        # # Q2  num    num     num
        # for b in range(embeddings.shape[0]):
        #     for i in range(len(grid[b])):
        #         for j in range(len(grid[b][i])):
        #             if j > i:
        #                 grid[b][i][j] = -np.inf
        # for b in range(embeddings.shape[0]):
        #     for row in range(len(grid[b])):
        #         grid[b][row] = softmax.softmax(np.array(grid[b][row]))
        # output = []
        # for b in range(embeddings.shape[0]):
        #     output.append([])  # New embeddings
        #     for i in range(len(key_embeddings[0])):
        #         output[-1].append(value_embeddings[b][0] * grid[b][i][0])
        #         # Multiply v_0 (matrix) by value in grid (scalar)
        #         for j in range(1, len(key_embeddings[0])):
        #             output[-1][-1] += value_embeddings[b][j] * grid[b][i][j]
        #             # Multiply next v_* by next value in grid
        # return np.array(output)
        scores = query_embeddings @ key_embeddings.transpose(0, 2, 1)
        scores /= np.sqrt(self.D_MODEL)
        mask = np.triu(
            np.ones(scores.shape[-2:], dtype=bool),
            k=1
        )
        scores = np.where(mask, -np.inf, scores)
        weights = softmax.softmax(scores)
        context = weights @ value_embeddings
        return context
