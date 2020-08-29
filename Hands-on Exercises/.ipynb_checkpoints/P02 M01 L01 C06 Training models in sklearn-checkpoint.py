

import numpy as np
import pandas as pd
import visuals as v
# Read the data
data = pd.read_csv('data/P02 M01 L01 C06.csv')

# Split the data into X and y
X = np.array(data[['x1', 'x2']])
y = np.array(data['y'])
#%%
# Logistic Regression

from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression()
classifier.fit(X,y)
v.plot_decision_boundary(X, y, classifier, 'Logistic Regression')

#%%
# Decision Tree

from sklearn.tree import DecisionTreeClassifier

classifier = DecisionTreeClassifier()
classifier.fit(X,y)
v.plot_decision_boundary(X, y, classifier, 'Decision Tree')

#%%
# Support Vector Machine

from sklearn.svm import SVC

classifier = SVC()
classifier.fit(X,y)
v.plot_decision_boundary(X, y, classifier, 'Support Vector Machine')

#%%
# Neural Network

from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.utils import to_categorical

# Building the model
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(2,)))
model.add(Dropout(.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(.1))
model.add(Dense(2, activation='softmax'))

# Compiling the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Training the model
features = X
target = np.array(to_categorical(y, 2))
model.fit(features, target, epochs=40, verbose=0)

# Plotting the decision boundary
v.plot_decision_boundary(X, y, model, 'Neural Network')
