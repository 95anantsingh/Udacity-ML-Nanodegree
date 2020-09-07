

# Imports
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
import pandas as pd
import numpy as np
import visuals as v

data = pd.read_csv('data/P02 M01 L01 C06.csv')

X = np.array(data[['x1', 'x2']])
y = np.array(data['y'])

# TODO: Uncomment one of the three classifiers, and hit "Test Run"
# to see the learning curve. Use these to answer the quiz below.

# %%
# Logistic Regression

estimator = LogisticRegression()
estimator.fit(X, y)

v.draw_learning_curves(X, y, estimator, 50, 'Logistic Regression')
v.plot_decision_boundary(X, y, estimator, 'Logistic Regression')

# %%
# Decision Tree

estimator = GradientBoostingClassifier()
estimator.fit(X, y)

v.draw_learning_curves(X, y, estimator, 50, 'Decision Tree')
v.plot_decision_boundary(X, y, estimator, 'Decision Tree')

# %%
# Support Vector Machine

estimator = SVC(kernel='rbf', gamma=1000)
estimator.fit(X, y)

v.draw_learning_curves(X, y, estimator, 50, 'Support Vector Machine')
v.plot_decision_boundary(X, y, estimator, 'Support Vector Machine')
