

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import visuals as v

# Read in the data.
data = np.asarray(pd.read_csv('data/P02 M01 L01 C07.csv'))

# Assign the features to the variable X, and the labels to the variable y.
X = data[:,0:2]
y = data[:,2]

# Use train test split to split your data
# Use a test size of 25% and a random state of 42
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state=42)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

acc= accuracy_score(y_test,y_pred)


# print and visuals
v.plot_decision_boundary(X,y,model, 'Decision Tree')
print(acc)