# Import necessary libraries for data handling, modeling, and plotting
import pandas as pd           # For data manipulation
import numpy as np            # For numerical operations
import matplotlib.pyplot as plt  # For plotting
import seaborn as sns         # For enhanced visualizations

import statsmodels.api as sm  # For regression analysis
from statsmodels.tsa.arima.model import ARIMA  # For time-series analysis
import scipy.stats as stats   # For hypothesis testing

# ===========================
# Step 1: Load the Normalized Dataset
# ===========================
# Read the normalized CSV file into a pandas DataFrame.
data = pd.read_csv('normalized_data.csv')
print("First 5 rows of the normalized dataset:")
print(data.head())

# ===========================
# Regression Analysis:
# Investigate the relationship among the variables
# ===========================
# Here we assume that 'Temperature' is our dependent variable,
# and the remaining numeric variables are used as predictors.

# Identify numeric columns in the dataset.
numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()

# Remove 'Temperature' from the predictors list (if present)
if 'Temperature' in numeric_cols:
    numeric_cols.remove('Temperature')

# Define the independent variables (X) and the dependent variable (y)
X = data[numeric_cols]
y = data['Temperature']

# Add a constant to the predictors to model the intercept
X_const = sm.add_constant(X)

# Fit a multiple linear regression model using Ordinary Least Squares (OLS)
model = sm.OLS(y, X_const)
results = model.fit()

# Print the regression summary which includes coefficients, p-values, R-squared, etc.
print("\nRegression Analysis Summary:")
print(results.summary())

# ===========================
# Time-Series Analysis:
# Forecast future global temperatures using an ARIMA model
# ===========================
# We assume that the Temperature data is ordered in time.
# For demonstration, we use an ARIMA(1,1,1) model to forecast the next 10 time periods.

# Extract the Temperature series from the dataset
temperature_series = data['Temperature']

# Fit an ARIMA model; adjust the (p, d, q) order as needed.
arima_model = ARIMA(temperature_series, order=(1, 1, 1))
arima_result = arima_model.fit()

# Forecast the next 10 time steps
forecast_steps = 10
forecast = arima_result.forecast(steps=forecast_steps)

print("\nARIMA Forecast for the next 10 time periods (Temperature):")
print(forecast)

# Plot the original temperature series along with the forecasted values
plt.figure(figsize=(10, 5))
plt.plot(temperature_series, label='Observed Temperature')
plt.plot(np.arange(len(temperature_series), len(temperature_series) + forecast_steps), 
         forecast, label='Forecast', color='red', marker='o')
plt.title("Temperature Forecast using ARIMA")
plt.xlabel("Time Index (Assumed Sequential Order)")
plt.ylabel("Normalized Temperature")
plt.legend()
plt.show()

# ===========================
# Hypothesis Testing:
# Evaluate the impact of human activities on climate change
# ===========================
# For demonstration, we assume that human impact has increased over time.
# We split the dataset into an "early" period and a "later" period,
# then perform an independent t-test to see if there is a significant difference in Temperature.

# Determine the midpoint of the dataset to split it into two periods.
split_index = len(data) // 2

# Define two groups based on the index: early and later periods.
group_early = data['Temperature'].iloc[:split_index]
group_later = data['Temperature'].iloc[split_index:]

# Perform an independent t-test comparing the two groups.
t_stat, p_value = stats.ttest_ind(group_early, group_later)

print("\nHypothesis Testing - T-test Results:")
print("T-statistic: {:.4f}".format(t_stat))
print("P-value: {:.4f}".format(p_value))

# Interpretation:
# A low p-value (typically < 0.05) suggests a significant difference between the two periods,
# which may be attributed to increased human activities affecting climate change.
