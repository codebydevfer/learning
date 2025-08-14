import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load Titanic dataset from your CSV
titanic = pd.read_csv('02-pandas-numpy/Day_06/Titanic-Dataset.csv')

# 2. Clean the data
titanic['Age'] = titanic['Age'].fillna(titanic['Age'].median())  # Fill missing ages with median

# Drop irrelevant columns (if they exist in your dataset)
cols_to_drop = ['Cabin', 'Ticket', 'Name']  
titanic = titanic.drop(columns=[col for col in cols_to_drop if col in titanic.columns])

# Drop rows where 'Embarked' is missing
titanic = titanic.dropna(subset=['Embarked'])

# 3. Summarize survival rates by class and gender
summary_df = (
    titanic.groupby(['Pclass', 'Sex'])['Survived']
    .mean()
    .mul(100)
    .reset_index()
    .rename(columns={'Survived': 'Survival Rate (%)'})
)

print(summary_df)

# 4. Plotting
plt.figure(figsize=(8, 6))
sns.barplot(data=summary_df, x='Pclass', y='Survival Rate (%)', hue='Sex')
plt.title("Titanic Survival Rates by Class and Gender", fontsize=16)
plt.ylabel("Survival Rate (%)")
plt.xlabel("Passenger Class")
plt.ylim(0, 100)
plt.legend(title="Gender")
plt.show()
