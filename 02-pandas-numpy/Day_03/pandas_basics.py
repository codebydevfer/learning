#Series

#1
import pandas as pd

a = [1, 7, 2]

myvar = pd.Series(a)

print(myvar)

#Creating Labels
#2
import pandas as pd

a = [1, 7, 2]

myvar = pd.Series(a, index = ["x", "y", "z"])

print(myvar)

print(myvar["y"])

#3
import pandas as pd

calories = {"day1": 420, "day2": 380, "day3": 390}

myvar = pd.Series(calories)

print(myvar)

#4
import pandas as pd

calories = {"day1": 420, "day2": 380, "day3": 390}

myvar = pd.Series(calories, index = ["day1", "day2"])

print(myvar)

#5
import pandas as pd

data = {
  "calories": [420, 380, 390],
  "duration": [50, 40, 45]
}

myvar = pd.DataFrame(data)

print(myvar)

#DataFrame

#1
import pandas as pd

data = {
  "calories": [420, 380, 390],
  "duration": [50, 40, 45]
}

#load data into a DataFrame object:
df = pd.DataFrame(data)

print(df)

#2
#Return row 0:

#refer to the row index:
print(df.loc[0])

#or

#use a list of indexes:
print(df.loc[[0, 1]])

#3
import pandas as pd

data = {
  "calories": [420, 380, 390],
  "duration": [50, 40, 45]
}

df = pd.DataFrame(data, index = ["day1", "day2", "day3"])

print(df) 

#4
#Return "day2":

#refer to the named index:
print(df.loc["day2"])

#Loading files into a DataFrame

import pandas as pd

df = pd.read_csv('data.csv')

print(df)

#Reading CSVs

#Load the file

import pandas as pd

df = pd.read_csv('data.csv')

print(df.to_string()) 

#use to_string() to print the entire DataFrame.

#Or print without the to_string()

import pandas as pd

df = pd.read_csv('data.csv')

print(df) 

#The number of rows returned is defined in Pandas option settings.
#You can check your system's maximum rows with the pd.options.display.max_rows statement.

import pandas as pd

print(pd.options.display.max_rows) 

#Increasing number of rows to be displayed

import pandas as pd

pd.options.display.max_rows = 9999

df = pd.read_csv('data.csv')

print(df) 

#Filtering

#1
import pandas as pd

data = {
  "name": ["Sally", "Mary", "John"],
  "age": [50, 40, 30],
  "qualified": [True, False, False]
}
df = pd.DataFrame(data)

newdf = df.filter(items=["name", "age"])

#Syntax

#dataframe.filter(items, like, regex, axis)




#Reference - W3Schools