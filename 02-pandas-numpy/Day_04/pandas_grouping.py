#Grouping

#1
import pandas as pd

data = {
  'co2': [95, 90, 99, 104, 105, 94, 99, 104],
  'model': ['Citigo', 'Fabia', 'Fiesta', 'Rapid', 'Focus', 'Mondeo', 'Octavia', 'B-Max'],
  'car': ['Skoda', 'Skoda', 'Ford', 'Skoda', 'Ford', 'Ford', 'Skoda', 'Ford']
}

df = pd.DataFrame(data)

print(df.groupby(["car"]).mean())

#Syntax

#dataframe.transform(by, axis, level, as_index, sort, group_keys, observed, dropna)

#Parameters

#https://www.w3schools.com/python/pandas/ref_df_groupby.asp

#value_counts

#1
import pandas as pd

# Create a sample Series
s = pd.Series(['apple', 'banana', 'apple', 'orange', 'banana', 'apple'])

# Count the occurrences of each unique value
counts = s.value_counts()
print(counts)

# Output:
# apple     3
# banana    2
# orange    1
# dtype: int64

# Count with normalization (percentages)
percentages = s.value_counts(normalize=True)
print(percentages)

# Output:
# apple     0.500000
# banana    0.333333
# orange    0.166667
# dtype: float64

#aggregation (mean, sum)

#Return the sum of each row:
#1
import pandas as pd

data = {
  "x": [50, 40, 30],
  "y": [300, 1112, 42]
}

df = pd.DataFrame(data)

x = df.aggregate(["sum"])

print(x)

#Syntax

#dataframe.aggregate(func, axis, args, kwargs)

#Parameters

#https://www.w3schools.com/python/pandas/ref_df_aggregate.asp

#pandas_plot() / matplotlib




#start project Spotify EDA







#Reference - W3Schools