# Import necessary libraries
import pandas as pd           # For data manipulation
import numpy as np            # For numerical operations
from sklearn.preprocessing import StandardScaler  # For data normalization

# -----------------------------
# Step 1: Load the Dataset with Custom Missing Value Indicators
# -----------------------------
# Specify additional missing value indicators: "Unknown" and "NAN"
missing_values = ["Unknown", "Unknown ", "NAN", "NAN "]

# Read the CSV file into a pandas DataFrame, converting specified strings to NaN.
data = pd.read_csv('climate_change_dataset.csv', na_values=missing_values)

# Display the first few rows to check the data structure.
print("First 5 rows of the dataset:")
print(data.head())

# -----------------------------
# Step 2: Data Inspection and Missing Values Check
# -----------------------------
# Get a summary of the dataset, including data types and non-null counts.
print("\nDataset information:")
print(data.info())

# Check for missing values in each column.
print("\nMissing values in each column:")
print(data.isnull().sum())

# -----------------------------
# Step 3: Handle Missing Data
# -----------------------------
# For numeric columns, fill missing values with the mean of the column.
# Identify numeric columns.
numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()

# Fill missing values for each numeric column with its mean.
for col in numeric_columns:
    data[col].fillna(data[col].mean(), inplace=True)

# Verify that missing values have been addressed in numeric columns.
print("\nMissing values after imputation for numeric columns:")
print(data[numeric_columns].isnull().sum())

# -----------------------------
# Step 4: Normalize the Data
# -----------------------------
# Normalize the numeric columns so that they have a mean of 0 and standard deviation of 1.
scaler = StandardScaler()

# Create a copy of the dataset for normalization.
data_normalized = data.copy()
data_normalized[numeric_columns] = scaler.fit_transform(data_normalized[numeric_columns])

# Display the first few rows of the normalized dataset.
print("\nFirst 5 rows of the normalized dataset:")
print(data_normalized.head())

# -----------------------------
# Step 5: Export the Normalized Data to a New CSV File
# -----------------------------
# Write the normalized dataset to a new CSV file named 'normalized_data.csv'
data_normalized.to_csv('normalized_data.csv', index=False)

print("\nNormalized data has been exported to 'normalized_data.csv'.")
