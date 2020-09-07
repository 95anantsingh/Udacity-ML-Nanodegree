# Import statements
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import visuals as v

# Read the data.
# data = np.asarray(pd.read_csv('data/P02 M01 L01 C07.csv', header=None))
data = np.asarray(pd.read_csv('data/P02 M01 L01 C07.csv'))

# Assign the features to the variable X, and the labels to the variable y.
X = data[:, 0:2]
y = data[:, 2]

# TODO: Create the decision tree model and assign it to the variable model.
# You won't need to, but if you'd like, play with hyperparameters such
# as max_depth and min_samples_leaf and see what they do to the decision
# boundary.
model = DecisionTreeClassifier(max_depth=6, min_samples_leaf=2,
                               random_state=42)

# TODO: Fit the model.
model.fit(X, y)

# TODO: Make predictions. Store them in the variable y_pred.
y_pred = model.predict(X)

# TODO: Calculate the accuracy and assign it to the variable acc.
acc = accuracy_score(y, y_pred)
print(acc * 100)

v.plot_decision_boundary(X, y, model, 'Decision Tree')
