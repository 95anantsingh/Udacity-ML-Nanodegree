

import pandas
import numpy as np
import visuals as v

data = pandas.read_csv('data/P02 M01 L01 C07.csv')

X = np.array(data[['x1','x2']])
y = np.array(data['y'])


# TODO: Define your classifier.
# Play with different values for these, from the options above.
# Hit 'Test Run' to see how the classifier fit your data.
# Once you can correctly classify all the points, hit 'Submit'.
# classifier = SVC(kernel = 'poly', degree = 1, gamma = 1, C = 1)

#%%
# Decision Tree

from sklearn.tree import DecisionTreeClassifier

classifier = DecisionTreeClassifier(max_depth=1000)
classifier.fit(X,y)
v.plot_decision_boundary(X, y, classifier, 'Decision Tree')

#%%
# Support Vector Machine
# kernel = Linear

from sklearn.svm import SVC

classifier = SVC(kernel='linear')
classifier.fit(X,y)
v.plot_decision_boundary(X, y, classifier, 'Support Vector Machine (linear)')

#%%
# Support Vector Machine
# kernel = poly

from sklearn.svm import SVC

classifier = SVC(kernel='poly', degree=2)
classifier.fit(X,y)
v.plot_decision_boundary(X, y, classifier, 'Support Vector Machine (poly)')

#%%
# Support Vector Machine
# kernel = rbf

from sklearn.svm import SVC

classifier = SVC(kernel='rbf', gamma=100,)
classifier.fit(X,y)
v.plot_decision_boundary(X, y, classifier, 'Support Vector Machine (rbf)')


