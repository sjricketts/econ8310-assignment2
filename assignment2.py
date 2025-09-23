# Our import statements for this problem
import pandas as pd
import numpy as np
import patsy as pt

import sklearn.tree as tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

data = pd.read_csv("https://raw.githubusercontent.com/dustywhite7/Econ8310/master/AssignmentData/assignment3.csv")

# drop string and datetime columns
columns_drop = ['id','DateTime']
data = data.drop(columns_drop, axis=1)

# Separate x and y variables
Y = data['meal']
X = data.drop('meal', axis = 1)

# Initializing the model
model = tree.DecisionTreeClassifier()

# Fit model
modelFit = model.fit(X,Y)

# Make predictions
forecast = modelFit.predict(X)

# convert to list
pred = forecast.tolist()