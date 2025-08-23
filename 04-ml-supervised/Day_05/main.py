#Data splitting

#Train/Test

#Train Test Split Procedure

#1 - Import all necessary libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

#2 - Load the Data Set

url = 'https://raw.githubusercontent.com/mGalarnyk/Tutorial_Data/master/King_County/kingCountyHouseData.csv'
df = pd.read_csv(url)
# Selecting columns I am interested in
columns = ['bedrooms','bathrooms','sqft_living','sqft_lot','floors','price']
df = df.loc[:, columns]
df.head(10)

#3 - Arrange Data Into Features and Target

features = ['bedrooms','bathrooms','sqft_living','sqft_lot','floors']
X = df.loc[:, features]
y = df.loc[:, ['price']]

#4 - Split Data Into Training and Testing Sets

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, train_size = .75)

#Creating and Training a Model in Scikit-Learn

#1 - Import the Model You Want to Use

from sklearn.tree import DecisionTreeRegressor

#2 - Make an Instance of the Model

reg = DecisionTreeRegressor(max_depth = 2, random_state = 0)

#3 - Train the Model on the Data

reg.fit(X_train, y_train)

#4 - Predict Labels of Unseen Test Data

# Predicting multiple observations
reg.predict(X_test[0:10])

X_test.head(1)

# predict 1 observation.
reg.predict(X_test.iloc[0].values.reshape(1,-1))


#cross-validation (KFold)

#1

[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

#2

Fold1: [0.5, 0.2]
Fold2: [0.1, 0.3]
Fold3: [0.4, 0.6]

#ex.

# Model1: Trained on Fold1 + Fold2, Tested on Fold3
# Model2: Trained on Fold2 + Fold3, Tested on Fold1
# Model3: Trained on Fold1 + Fold3, Tested on Fold2

#Cross-Validation API

#1

kfold = KFold(3, True, 1)

#2


# enumerate splits
for train, test in kfold.split(data):
	print('train: %s, test: %s' % (train, test))

#3

# scikit-learn k-fold cross-validation
from numpy import array
from sklearn.model_selection import KFold
# data sample
data = array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
# prepare cross validation
kfold = KFold(3, True, 1)
# enumerate splits
for train, test in kfold.split(data):
	print('train: %s, test: %s' % (data[train], data[test]))
 
#4
 
# train: [0.1 0.4 0.5 0.6], test: [0.2 0.3]
# train: [0.2 0.3 0.4 0.6], test: [0.1 0.5]
# train: [0.1 0.2 0.3 0.5], test: [0.4 0.6]






#Reference - https://builtin.com/data-science/train-test-split
#Reference - https://machinelearningmastery.com/k-fold-cross-validation/