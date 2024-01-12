# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 23:32:02 2023

@author: harsh
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


# Set working directory
working_directory = "C:/Users/harsh/OneDrive/Documents/Healthcare/Final/archive/"

# Define file paths
full_grouped_path = f"{working_directory}/full_grouped.csv"
country_wise_latest_path = f"{working_directory}/country_wise_latest.csv"
usa_county_wise_path = f"{working_directory}/usa_county_wise.csv"
usa_county_wise_clean_path = f"{working_directory}/usa_count_wise_clean.csv"

# Read data
full_grouped = pd.read_csv(full_grouped_path)
country_wise_latest = pd.read_csv(country_wise_latest_path)
usa_county_wise = pd.read_csv(usa_county_wise_path)

# Check missing values
print("Missing values in full_grouped:")
print(full_grouped.isnull().sum())

print("Missing values in country_wise_latest:")
print(country_wise_latest.isnull().sum())

print("Missing values in usa_county_wise:")
print(usa_county_wise.isnull().sum())

# Clean USA county data
usa_county_wise_clean = usa_county_wise.drop(columns=["FIPS", "Admin2"])

# Save cleaned data
usa_county_wise_clean.to_csv(usa_county_wise_clean_path, index=False)


###################################

# Load the dataset
file_path = country_wise_latest_path

data = pd.read_csv(file_path)

# Drop the 'Country/Region' and 'WHO Region' columns
predictors = data.drop(['Country/Region', 'WHO Region', 'Confirmed'], axis=1)

# Replace infinities with NaN
predictors.replace([np.inf, -np.inf], np.nan, inplace=True)

# Impute the NaN values with the median
predictors['Deaths / 100 Recovered'].fillna(predictors['Deaths / 100 Recovered'].median(), inplace=True)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    predictors, 
    data['Confirmed'], 
    test_size=0.2, 
    random_state=42
)

# Initialize the Random Forest Regressor
rf = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
rf.fit(X_train, y_train)

# Predict on the test set
y_pred = rf.predict(X_test)

# Calculate the root mean square error (RMSE)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2_score = rf.score(X_test, y_test)

print(f"RMSE: {rmse}")
print(f"R^2 Score: {r2_score}")


# Create a DataFrame with actual and predicted values
results_df = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_pred
})

y_pred= np.round(y_pred).astype(int)

result_file_path=f"{working_directory}/model_predictions.csv"

# Save the DataFrame to a CSV file
results_df.to_csv(result_file_path,index=False)

