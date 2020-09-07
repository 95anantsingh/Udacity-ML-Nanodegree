# Import statements
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pandas as pd

# Read the data.
data = pd.read_csv('data/P02 M01 L01 C07.csv')
# Assign the features to the variable X, and the labels to the variable y.
X = data[['x1', 'x2']]
y = data['y']

# TODO: Create the model and assign it to the variable model.
# Find the right parameters for this model to achieve 100% accuracy on the dataset.
model = SVC(kernel='rbf', gamma=27, random_state=0)

# TODO: Fit the model.
model.fit(X, y)
# TODO: Make predictions. Store them in the variable y_pred.
y_pred = model.predict(X)

# TODO: Calculate the accuracy and assign it to the variable acc.
acc = accuracy_score(y, y_pred)