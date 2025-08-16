#Linear Regression

#1
import matplotlib.pyplot as plt

x = [5,7,8,7,2,17,2,9,4,11,12,9,6]
y = [99,86,87,88,111,86,103,87,94,78,77,85,86]

plt.scatter(x, y)
plt.show()

#2 - Import scipy and draw the line of Linear Regression:

import matplotlib.pyplot as plt
from scipy import stats

x = [5,7,8,7,2,17,2,9,4,11,12,9,6]
y = [99,86,87,88,111,86,103,87,94,78,77,85,86]

slope, intercept, r, p, std_err = stats.linregress(x, y)

def myfunc(x):
  return slope * x + intercept

mymodel = list(map(myfunc, x))

plt.scatter(x, y)
plt.plot(x, mymodel)
plt.show()

#2 Explanation

#Create the arrays that represent the values of the x and y axis:

x = [5,7,8,7,2,17,2,9,4,11,12,9,6]
y = [99,86,87,88,111,86,103,87,94,78,77,85,86]

#Execute a method that returns some important key values of Linear Regression:

slope, intercept, r, p, std_err = stats.linregress(x, y)

#Create a function that uses the slope and intercept values to return a new value. This new value represents where on the y-axis the corresponding x value will be placed

def myfunc(x):
  return slope * x + intercept

#Run each value of the x array through the function. This will result in a new array with new values for the y-axis

mymodel = list(map(myfunc, x))

#Draw the original scatter plot

plt.scatter(x, y)

#Draw the line of linear regression

plt.plot(x, mymodel)

#Display the diagram

plt.show()

#3 - It is important to know how the relationship between the values of the x-axis and the values of the y-axis is, if there are no relationship the linear regression can not be used to predict anything. This relationship - the coefficient of correlation - is called r. The r value ranges from -1 to 1, where 0 means no relationship, and 1 (and -1) means 100% related. Python and the Scipy module will compute this value for you, all you have to do is feed it with the x and y values.

from scipy import stats

x = [5,7,8,7,2,17,2,9,4,11,12,9,6]
y = [99,86,87,88,111,86,103,87,94,78,77,85,86]

slope, intercept, r, p, std_err = stats.linregress(x, y)

print(r)

#4 - Predict Future Values

#Predicting the speed of a 10 years old car. To do so, we need the same myfunc() function from the example above

from scipy import stats

x = [5,7,8,7,2,17,2,9,4,11,12,9,6]
y = [99,86,87,88,111,86,103,87,94,78,77,85,86]

slope, intercept, r, p, std_err = stats.linregress(x, y)

def myfunc(x):
  return slope * x + intercept

speed = myfunc(10)

print(speed)

#Cost Function - https://builtin.com/machine-learning/cost-function

#mae

def mae(predictions, targets):
    # Retrieving number of samples in dataset
    samples_num = len(predictions)
   
    # Summing absolute differences between predicted and expected values
    accumulated_error = 0.0
    for prediction, target in zip(predictions, targets):
        accumulated_error += np.abs(prediction - target)
       
    # Calculating mean
    mae_error = (1.0 / samples_num) * accumulated_error
   
    return mae_error

#mse

def mse(predictions, targets):
    # Retrieving number of samples in dataset
    samples_num = len(predictions)
   
    # Summing square differences between predicted and expected values
    accumulated_error = 0.0
    for prediction, target in zip(predictions, targets):
        accumulated_error += (prediction - target)**2
       
    # Calculating mean and dividing by 2
    mse_error = (1.0 / (2*samples_num)) * accumulated_error
   
    return mse_error

#mae or mse?

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Linear model

def predict(x, parameters):

    return parameters["w"] * x + parameters["b"]

# Load data from .csv
df_data = pd.read_csv("cracow_apartments.csv", sep=",")

# Used features and target value
features = ["size"]
target = ["price"]

# Slice Dataframe to separate feature vectors and target value
X, y = df_data[features].to_numpy(), df_data[target].to_numpy()

# Parameter sets
orange_parameters = {'b': 200, 'w': np.array([3.0])}
lime_parameters = {'b': -160, 'w': np.array([12.0])}

# Make prediction for every data sample
orange_pred = [predict(x, orange_parameters) for x in X]
lime_pred = [predict(x, lime_parameters) for x in X]

# Model error
mse_orange_error = mse(orange_pred, y)
mse_lime_error = mse(lime_pred, y)

#Gradient Descent - https://www.geeksforgeeks.org/data-science/what-is-gradient-descent/

#Pseudo Code

# t ← 0
# max_iterations ← 1000
# w, b ← initialize randomly

# while t < max_iterations do
#     t ← t + 1
#     w_t+1 ← w_t − γ ∇w_t
#     b_t+1 ← b_t − γ ∇b_t
# end

# Here:
# max_iterations is the number of iteration we want to do to update our parameter 
# W,b are the weights and bias parameter 
# γ is the learning parameter  

#Implementing linear regression with scikit-learn - https://www.geeksforgeeks.org/machine-learning/python-linear-regression-using-sklearn/

#Step 1 - Importing all the required libraries 

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

#Step 2 - Reading the dataset

df = pd.read_csv('https://www.kaggle.com/datasets/dftow001/bottle-csv')
df_binary = df[['Salnty', 'T_degC']]

# Taking only the selected two attributes from the dataset
df_binary.columns = ['Sal', 'Temp']
#display the first 5 rows
df_binary.head()

#Step 3 - Exploring the data scatter 

#plotting the Scatter plot to check relationship between Sal and Temp
sns.lmplot(x ="Sal", y ="Temp", data = df_binary, order = 2, ci = None)
plt.show()

#Step 4 - Data cleaning 

# Eliminating NaN or missing input numbers
df_binary.fillna(method ='ffill', inplace = True)

#Step 5 - Training our model

X = np.array(df_binary['Sal']).reshape(-1, 1)
y = np.array(df_binary['Temp']).reshape(-1, 1)

# Separating the data into independent and dependent variables
# Converting each dataframe into a numpy array 
# since each dataframe contains only one column
df_binary.dropna(inplace = True)

# Dropping any rows with Nan values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

# Splitting the data into training and testing data
regr = LinearRegression()

regr.fit(X_train, y_train)
print(regr.score(X_test, y_test))

#Step 6 - Exploring our results 

y_pred = regr.predict(X_test)
plt.scatter(X_test, y_test, color ='b')
plt.plot(X_test, y_pred, color ='k')

plt.show()
# Data scatter of predicted values

#Step 7 - Working with a smaller dataset 

df_binary500 = df_binary[:][:500]
  
# Selecting the 1st 500 rows of the data
sns.lmplot(x ="Sal", y ="Temp", data = df_binary500,
                               order = 2, ci = None)

#First 500 rows

df_binary500.fillna(method ='fill', inplace = True)

X = np.array(df_binary500['Sal']).reshape(-1, 1)
y = np.array(df_binary500['Temp']).reshape(-1, 1)

df_binary500.dropna(inplace = True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

regr = LinearRegression()
regr.fit(X_train, y_train)
print(regr.score(X_test, y_test))

#

y_pred = regr.predict(X_test)
plt.scatter(X_test, y_test, color ='b')
plt.plot(X_test, y_pred, color ='k')

plt.show()

#Step 8 - Evaluation Metrics For Regression

from sklearn.metrics import mean_absolute_error,mean_squared_error

mae = mean_absolute_error(y_true=y_test,y_pred=y_pred)
#squared True returns MSE value, False returns RMSE value.
mse = mean_squared_error(y_true=y_test,y_pred=y_pred) #default=True
rmse = mean_squared_error(y_true=y_test,y_pred=y_pred,squared=False)

print("MAE:",mae)
print("MSE:",mse)
print("RMSE:",rmse)













#Reference - W3Schools
#Reference - Built In
#Reference - GeeksForGeeks