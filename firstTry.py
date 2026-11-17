import numpy as np
import string
import matplotlib.pyplot as plt
import sys

import tokeniser


def initialize_parameters(input_size, hidden_size, output_size):
    np.random.seed(42)
    parameters = {
        "W1": np.random.randn(hidden_size, input_size) * 0.01,
        "b1": np.zeros((hidden_size, 1)),
        "W2": np.random.randn(output_size, hidden_size) * 0.01,
        "b2": np.zeros((output_size, 1))
    }
    return parameters


def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))


def relu(Z):
    return np.maximum(0, Z)


def relu_derivative(Z):
    return (Z > 0).astype(int)


def forward_propagation(X, parameters):
    W1, b1, W2, b2 = (
        parameters["W1"],
        parameters["b1"],
        parameters["W2"],
        parameters["b2"]
    )

    Z1 = np.dot(W1, X) + b1
    A1 = relu(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)

    cache = {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}
    return A2, cache


def compute_cost(Y, A2):
    m = Y.shape[1]
    cost = -np.sum(Y * np.log(A2) + (1 - Y) * np.log(1 - A2)) / m
    if 0 in 1 - A2:
        raise ValueError
    return np.squeeze(cost)


def backward_propagation(X, Y, parameters, cache):
    m = X.shape[1]
    W2 = parameters["W2"]

    dZ2 = cache["A2"] - Y
    dW2 = np.dot(dZ2, cache["A1"].T) / m
    db2 = np.sum(dZ2, axis=1, keepdims=True) / m

    dZ1 = np.dot(W2.T, dZ2) * relu_derivative(cache["Z1"])
    dW1 = np.dot(dZ1, X.T) / m
    db1 = np.sum(dZ1, axis=1, keepdims=True) / m

    grads = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}
    return grads


def update_parameters(parameters, grads, learning_rate):
    for key in parameters.keys():
        parameters[key] -= learning_rate * grads["d" + key]
    return parameters


def train_neural_network(X, Y,
                         input_size, hidden_size, output_size,
                         epochs=1000, learning_rate=0.01):
    parameters = initialize_parameters(input_size, hidden_size, output_size)
    costs = np.zeros(epochs)

    for i in range(epochs):
        A2, cache = forward_propagation(X, parameters)
        cost = compute_cost(Y, A2)
        grads = backward_propagation(X, Y, parameters, cache)
        # parameters = update_parameters(parameters, grads, learning_rate)
        parameters = update_parameters(parameters, grads, cost/1000)
        costs[i] = cost

        if i % 10 == 0:
            print(f"Epoch {i}: Cost = {cost}")

    return parameters, costs


def main():
    tokeniser_object = tokeniser.Tokeniser(string.printable[:-3])
    with open("training_data.txt") as file:
        inputs = tokeniser_object.tokenise(np.array(file.readlines()))
    print("Tokenised")
    input_vectors = [[] for _ in range(32)]
    targets = [[]]
    for i in inputs:
        so_far = [0 for _ in range(32)]
        for idx, token in enumerate(i[:-1]):
            so_far.append(token)
            so_far.pop(0)
            for j in range(32):
                input_vectors[j].append(so_far[j])
            targets[0].append(i[idx+1] / tokeniser_object.num_tokens)
    input_vectors = np.array(input_vectors)
    targets = np.array(targets)
    np.savetxt(sys.stdout, input_vectors)
    print(input_vectors.shape)
    print(targets.shape)
    print("Training data formatted")
    trained_parameters, costs = train_neural_network(
        input_vectors, targets,
        input_size=32, hidden_size=100, output_size=1,
        epochs=1_000_000, learning_rate=0.01
    )
    plt.plot(costs)
    plt.show()
    text = input("Enter first part: ")
    vector = tokeniser_object.tokenise(np.array([text]))[0].tolist()
    while len(vector) < 32:
        vector.insert(0, 0)
    vector = np.array([[x] for x in vector])
    print((forward_propagation(vector, trained_parameters)[0][0]))
    print(tokeniser_object.tokenToString(int(
        forward_propagation(vector, trained_parameters)[0][0][0]
        * tokeniser_object.num_tokens)))


if __name__ == "__main__":
    main()
