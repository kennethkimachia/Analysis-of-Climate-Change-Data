# Import necessary libraries for data manipulation and visualization
import pandas as pd           # For data manipulation
import numpy as np            # For numerical operations
import matplotlib.pyplot as plt  # For plotting
import seaborn as sns         # For enhanced visualizations

# -----------------------------
# Step 1: Load the Normalized Dataset
# -----------------------------
# Read the normalized CSV file into a pandas DataFrame.
data_normalized = pd.read_csv('normalized_data.csv')

# Display the first few rows to verify the data
print("First 5 rows of the normalized dataset:")
print(data_normalized.head())

# -----------------------------
# Step 2: Calculate Basic Descriptive Statistics
# -----------------------------
# Identify numeric columns in the dataset.
numeric_columns = data_normalized.select_dtypes(include=[np.number]).columns.tolist()

# Create a dictionary to store statistics for each variable.
stats = {}

# Loop through each numeric column and calculate mean, median, and variance.
for col in numeric_columns:
    mean_val = data_normalized[col].mean()      # Calculate mean
    median_val = data_normalized[col].median()    # Calculate median
    variance_val = data_normalized[col].var()     # Calculate variance
    stats[col] = {'mean': mean_val, 'median': median_val, 'variance': variance_val}

# Print the basic statistics for each numeric variable.
print("\nBasic Statistics for each numeric variable:")
for col, stat in stats.items():
    print(f"{col}: Mean = {stat['mean']:.4f}, Median = {stat['median']:.4f}, Variance = {stat['variance']:.4f}")

# -----------------------------
# Step 3: Visualize Trends One by One
# -----------------------------
# Set a visual style for the plots using seaborn.
sns.set(style="whitegrid")

# Loop over each numeric column and create an individual plot.
# The plt.show() call is blocking, so the next plot will only appear after you close the current window.
for col in numeric_columns:
    plt.figure()  # Create a new figure for each variable
    plt.plot(data_normalized.index, data_normalized[col], marker='o', linestyle='-', markersize=3)
    plt.title(f"Trend of {col}")  # Title for the subplot
    plt.xlabel("Index (Assumed Time Order)")
    plt.ylabel(col)
    plt.tight_layout()  # Adjust subplot spacing
    plt.show()  # Display the plot. This call blocks until you close the plot window.
