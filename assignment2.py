import pandas as pd
import numpy as np
from xgboost import XGBClassifier

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

# model
model = XGBClassifier(n_estimators=15, max_depth=3,learning_rate=0.5, objective='binary:logistic', random_state=42)

# fit
modelFit = model.fit(X, Y)

# Make predictions based on the testing x values
forecast = modelFit.predict(xt)

# convert to list
pred = forecast.tolist()