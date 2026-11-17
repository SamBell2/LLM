import numpy as np

import embeddings
import outputlayer
import transformer


class GPT:
    def __init__(self, D_MODEL: int, CONTEXT_LENGTH: int, BATCH_SIZE: int,
                 NUM_HEADS: int, D_HEAD: int, NUM_TRANSFORMERS: int,
                 VOCABULARY_SIZE: int,
                 vocab_embeddings: np.ndarray | None = None,
                 position_embeddings: np.ndarray | None = None,
                 wqs: np.ndarray | None = None,
                 wks: np.ndarray | None = None,
                 wvs: np.ndarray | None = None,
                 w1s: np.ndarray | None = None,
                 w2s: np.ndarray | None = None,
                 b1s: np.ndarray | None = None,
                 b2s: np.ndarray | None = None,
                 beta1s: np.ndarray | None = None,
                 gamma1s: np.ndarray | None = None,
                 beta2s: np.ndarray | None = None,
                 gamma2s: np.ndarray | None = None,
                 output_matrix: np.ndarray | None = None):
        self.embedder = embeddings.Embeddings(VOCABULARY_SIZE, D_MODEL,
                                              CONTEXT_LENGTH, vocab_embeddings,
                                              position_embeddings)
        self.blocks = []
        for i in range(NUM_TRANSFORMERS):
            if wqs is None:
                trans = transformer.Transformer(D_MODEL, D_HEAD)
            else:
                trans = transformer.Transformer(D_MODEL, D_HEAD,
                                                wqs[i], wks[i], wvs[i], w1s[i],
                                                w2s[i], b1s[i], b2s[i],
                                                beta1s[i], gamma1s[i],
                                                beta2s[i], gamma2s[i])
            self.blocks.append(trans)
        self.ol = outputlayer.OutputLayer(D_MODEL,
                                          VOCABULARY_SIZE,
                                          output_matrix)

    def __call__(self, intputs: np.ndarray) -> None:
        x = self.embedder(intputs)
        for block in self.blocks:
            x = block(x)
        x = self.ol(x)
        return x
