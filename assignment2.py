# Our import statements for this problem
import pandas as pd
import numpy as np
import patsy as pt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# read in data
data = pd.read_csv("https://raw.githubusercontent.com/dustywhite7/Econ8310/master/AssignmentData/assignment3.csv")
test_data = pd.read_csv("https://raw.githubusercontent.com/dustywhite7/Econ8310/master/AssignmentData/assignment3test.csv")

# clean data
columns_drop = ['id','DateTime']
data = data.drop(columns_drop, axis=1)

test_columns_drop = ['id','DateTime']
test_data = test_data.drop(test_columns_drop, axis=1)

# data
Y = data['meal']
X = data.drop('meal', axis = 1)
yt = test_data['meal']
xt = test_data.drop('meal', axis = 1)

# generate random forest model
model = RandomForestClassifier(n_estimators=100, n_jobs = -1, max_depth=10)

# fit model
modelFit = model.fit(X,Y)

# Make predictions
forecast = modelFit.predict(xt)

# convert to list
pred = forecast.tolist()