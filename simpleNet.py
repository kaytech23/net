import numpy as np
import math


np.random.seed(1)


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def sigmoid_array(arr):
    return np.array([sigmoid(row) for row in arr])


def get_xor(row):
    if row[0] != row[1]:
        return 1
    else:
        return 0


def generate_row():
    row = np.random.random_integers(0, 1, 2)
    return [row, get_xor(row)]


def generate_training_set():
    input_data = []
    for _ in range(10):
        input_data.append(generate_row())
    return np.array(input_data)


training_set = generate_training_set()
print(training_set)

input_L = 2
h1_L = 5
output_L = 1
bias = np.random.normal(output_L)

h1_layer = {"Weight": np.random.normal(0, 0.5, [input_L, h1_L]),
            "Bias": np.random.normal([input_L, h1_L])}

output_layer = {"Weight": np.random.normal(0, 0.5, [h1_L, output_L]),
                "Bias": np.random.normal([h1_L, output_L])}

for row in training_set:
    input = row[0]
    y = np.matmul(input, h1_layer["Weight"])
    print(y)
    y = sigmoid_array(y)
    print(y)
    y = np.matmul(y, output_layer["Weight"]) + bias
    print(y)
    break

    #print(y, ":", row[1])
    #print(input)
    #print(y[0], ":", row[1])
    #break

