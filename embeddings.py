import numpy as np


class Embeddings:
    def __init__(self,
                 VOCABULARY_SIZE: int,
                 D_MODEL: int,
                 CONTEXT_LENGTH: int,
                 embeddings: np.ndarray | None = None,
                 position_embeddings: np.ndarray | None = None):
        if embeddings is None:
            self.embeddings = np.random.randn(VOCABULARY_SIZE, D_MODEL) * 0.02
            # * 0.02 is because its proven to work better
        else:
            self.embeddings = embeddings

        if position_embeddings is None:
            self.position_embeddings = np.random.randn(
                CONTEXT_LENGTH,
                D_MODEL
            ) * 0.01
        else:
            self.position_embeddings = position_embeddings

    def __call__(self, tokens):
        x = self.embeddings[tokens]
        x += self.position_embeddings[:tokens.shape[-1]]
        return x
