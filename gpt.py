import numpy as np

import tokeniser
import embeddings
import outputlayer
import transformer


class GPT:
    def __init__(self, D_MODEL: int, CONTEXT_LENGTH: int, BATCH_SIZE: int,
                 NUM_HEADS: int, D_HEAD: int, NUM_TRANSFORMERS: int,
                 VOCABULARY_SIZE: int,
                 vocab_embeddings: np.ndarray | None = None,
                 position_embeddings: np.ndarray | None = None):
        self.embedder = embeddings.Embeddings(VOCABULARY_SIZE, D_MODEL,
                                              CONTEXT_LENGTH, vocab_embeddings,
                                              position_embeddings)
        self.blocks = []
        for

    def __call__(self, intputs: np.ndarray) -> None:
        next_token = 0
        block = transformer.Transformer(BETA, GAMMA, D_MODEL, D_HEAD)
        ol = outputlayer.OutputLayer(D_MODEL, tokenizer.num_tokens)
        print(text[0], end=' ')
        while next_token != "[EOS]":
            tokens = tokenizer(text)
            # print(tokens.shape)
            x = embedder(tokens)
            for _ in range(NUM_TRANSFORMERS):
                x = block(x)
            x = ol(x)
            probabilities = softmax.softmax(x)[:, -1, :]
            indexes = []
            for batch in probabilities:
                indexes.append(np.random.choice(
                    range(tokenizer.num_tokens),
                    p=batch
                ))
                print(batch[indexes[-1]])
            # indexes = probabilities.argmax(axis=-1)
            # print(indexes.shape)
            next_token = tokenizer.tokenToString(indexes[0])
            text[0] = text[0] + ' ' + next_token
            print(next_token, end=' ')

        with open("training_data_shakespeare.txt") as file:
            text = file.read()
        print("Read training data")
        tokenizer = tokeniser.Tokeniser(text, CONTEXT_LENGTH)
        print("vocab built")
        # print(tokenizer._str_to_int)
        tokens = tokenizer(np.array(["Hail"]))
        # print(tokens)
        for token in tokens[0]:
            print(f"{token} = {repr(tokenizer.tokenToString(token))}")


if __name__ == "__main__":
    main()
