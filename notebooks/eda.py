# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

"""
import kagglehub

# Download latest version
path = kagglehub.dataset_download("jeleeladekunlefijabi/ship-performance-clustering-dataset")

print("Path to dataset files:", path)
"""

import pandas as pd
df = pd.read_csv(f"{path}/Ship_Performance_Dataset.csv")

df.sample(5)

df.describe(include='object')

print(f"{df.Weather_Condition.unique()=}")
print(f"{df.Maintenance_Status.unique()=}")
print(f"{df.Ship_Type.unique()=}")

df.describe(exclude="object")

print(df.select_dtypes(include='object').isnull().sum())

print(df.select_dtypes(exclude='object').isnull().sum())

# +
import seaborn as sns
import matplotlib.pyplot as plt
# Create a figure and a grid of subplots
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Flatten the array of axes for easy iteration
axes = axes.flatten()

# Loop through the object columns in the DataFrame
for i, col in enumerate(df.select_dtypes(include='object').columns.tolist()[1:]):
    sns.barplot(x=df[col].value_counts().index, y=df[col].value_counts().values, ax=axes[i])
    axes[i].set_title(f'Distribution of {col.replace("_", " ").title()}', size=15)
    axes[i].set_xlabel(f'{col.replace("_", " ").title()}')
    axes[i].set_ylabel('Frequency')

# Adjust layout
plt.tight_layout()
plt.show()


# +
# Create a figure and a grid of subplots
plt.figure(figsize=(20, 10))

# Loop through the object columns in the DataFrame
for i, col in enumerate(df.select_dtypes(exclude='object').columns.tolist()[1:]):
    plt.subplot(3, 4, i+1)
    sns.histplot(data=df[col], kde=True)
    plt.title(f'Distribution of {col.replace("_", " ").title()}', size=15)
    plt.xlabel(f'{col.replace("_", " ").title()}')
    plt.ylabel('Frequency')

# Adjust layout
plt.tight_layout()
plt.show()


# +
df.select_dtypes(exclude='object').corr()
plt.figure(figsize=(14,10))

sns.heatmap(df.select_dtypes(exclude='object').corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

# -


