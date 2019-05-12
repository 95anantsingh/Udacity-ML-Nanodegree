
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import learning_curve


def plot_decision_boundary(X, y, model, name):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])

    # For neural network training, the output contains probability of all classes
    if len(Z.shape) > 1 and Z.shape[1] > 1:
        Z = np.argmax(Z, axis=1)

    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.2, cmap=ListedColormap(('red', 'blue')))
    for i, j in enumerate(np.unique(y)):
        plt.scatter(X[y == j, 0], X[y == j, 1],
                    c=ListedColormap(('red', 'blue'))(i), label=j)

    # plt.rcParams['figure.facecolor'] = 'white'
    # plt.rcParams['figure.facecolor'] = 'white'
    plt.title(name)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.show()


def draw_learning_curves(X, y, estimator, num_trainings, name):

    # It is good to randomize the data before drawing Learning Curves

    np.random.seed(55)
    permutation = np.random.permutation(y.shape[0])
    X2 = X[permutation, :]
    y2 = y[permutation]

    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X2, y2, cv=None, n_jobs=1, train_sizes=np.linspace(.1, 1.0, num_trainings))

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.grid()

    plt.title("Learning Curves from " + name)
    plt.xlabel("Training examples")
    plt.ylabel("Error")

    plt.plot(1-train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(1-test_scores_mean, 'o-', color="b",
             label="Cross-validation score")

    plt.legend(loc="best")
    plt.show()