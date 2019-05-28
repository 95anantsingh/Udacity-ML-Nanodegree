import numpy as np
import visuals as v
import pandas as pd

# Setting the random seed, feel free to change it and see different solutions.
np.random.seed(42)


def stepFunction(t):
    if t >= 0:
        return 1
    return 0


def prediction(X, W, b):
    return stepFunction((np.matmul(X, W) + b)[0])


# TODO: Fill in the code below to implement the perceptron trick.
# The function should receive as inputs the data X, the labels y,
# the weights W (as an array), and the bias b,
# update the weights and bias W, b, according to the perceptron algorithm,
# and return W and b.
def perceptronStep(X, y, W, bias, learn_rate=0.01):
    # Fill in code
    for i in range(len(X)):
        yh = prediction(X[i], W, bias)
        if yh != y[i]:
            if yh > y[i]:
                W[0] = W[0] - learn_rate * X[i][0]
                W[1] = W[1] - learn_rate * X[i][1]
                bias = bias - learn_rate
            else:
                W[1] = W[0] + learn_rate * X[i][0]
                W[1] = W[1] + learn_rate * X[i][1]
                bias = bias + learn_rate

    return W, bias


# This function runs the perceptron algorithm repeatedly on the dataset,
# and returns a few of the boundary lines obtained in the iterations,
# for plotting purposes.
# Feel free to play with the learning rate and the num_epochs,
# and see your results plotted below.
def trainPerceptronAlgorithm(X, y, learn_rate=0.01, num_epochs=25):
    x_min, x_max = min(X.T[0]), max(X.T[0])
    y_min, y_max = min(X.T[1]), max(X.T[1])
    W = np.array(np.random.rand(2, 1))
    b = np.random.rand(1)[0] + x_max
    # These are the solution lines that get plotted below.
    boundary_lines = []
    for i in range(num_epochs):
        # In each epoch, we apply the perceptron step.
        W, b = perceptronStep(X, y, W, b, learn_rate)
        boundary_lines.append([W[0], W[1], b])
    return boundary_lines


# Load the data
data = pd.read_csv('data/P03 M01 L02 C09.csv')
X = data.iloc[:, 0:2].values
y = data.iloc[:, 2].values

# Train and Plot
b = trainPerceptronAlgorithm(X, y, num_epochs=25)
v.plot_decision_boundaries(X, y, b, list(range(0, 25, 8)), 'Perceptron')
