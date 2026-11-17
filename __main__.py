import numpy as np
from numpy.dtypes import StringDType

import tokeniser
import embeddings
import outputlayer
import transformer
import softmax


TRAINING_TEXT_PATH = "training_data_shakespeare copy.txt"
with open(TRAINING_TEXT_PATH) as f:
    TRAINING_TEXT: str = f.read()

# Model dimension
D_MODEL: int = 4
# Number of tokens model looks at to predict next one
CONTEXT_LENGTH: int = 8
# Number of statements processes simulateously in training
BATCH_SIZE: int = 1  # (simpler)
# Number of attention heads to use
NUM_HEADS: int = 1
# Size of vector handled by each head
D_HEAD: int = D_MODEL // NUM_HEADS
# Number of transformer blocks to use
NUM_TRANSFORMERS: int = 4
# Layer normalization parameters
BETA: float = 0
GAMMA: float = 1


def sample(tokens, num_samples):
    length = len(tokens)
    samples = []
    for _ in range(num_samples):
        start = np.random.randint(0, length-CONTEXT_LENGTH)
        end = start + CONTEXT_LENGTH
        section = tokens[start:end+1]
        samples.append(section)
    return np.array(samples)


def log_softmax(x):
    # assumes x is a vector
    x_max = np.max(x)
    return x - x_max - np.log(np.sum(np.exp(x - x_max)))


def cross_entropy(y_hat, y_true):
    return -log_softmax(y_hat)[y_true]


def train():
    tokenizer = tokeniser.Tokeniser(CONTEXT_LENGTH, TRAINING_TEXT)
    samples = sample(tokenizer._tokeniseString(TRAINING_TEXT), 2)
    embedder = embeddings.Embeddings(tokenizer.num_tokens, D_MODEL,
                                     CONTEXT_LENGTH)
    block = transformer.Transformer(BETA, GAMMA, D_MODEL, D_HEAD)
    ol = outputlayer.OutputLayer(D_MODEL, tokenizer.num_tokens)
    print(samples.shape)
    x = embedder(samples[:, :-1])
    for _ in range(NUM_TRANSFORMERS):
        x = block(x)
    x = ol(x)
    for batch in range(BATCH_SIZE):
        for i in range(CONTEXT_LENGTH):
            correct_token = samples[batch][i+1]
            ce_loss = cross_entropy(x[batch][i], correct_token)
            # ce_loss = -np.log(probabilities[batch][i][correct_token])
            print(ce_loss)


def main() -> None:
    # train()
    next_token = 0
    text = np.array([input("Enter prompt:\n > ")], dtype=StringDType())
    tokenizer = tokeniser.Tokeniser(CONTEXT_LENGTH, TRAINING_TEXT)
    embedder = embeddings.Embeddings(tokenizer.num_tokens, D_MODEL,
                                     CONTEXT_LENGTH)
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
