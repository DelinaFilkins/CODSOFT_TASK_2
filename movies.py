#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load your movies and rating dataset
# Replace 'your_movie_rating_dataset.csv' with the actual path to your dataset
df = pd.read_csv('ratings_small.csv')

# Select relevant features
selected_features = ['movieId']

# Drop rows with missing values for simplicity
df = df[selected_features + ['rating']].dropna()

# Split the dataset into features and target variable
X = df.drop('rating', axis=1)
y = df['rating']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared:", r2)


# In[ ]:




