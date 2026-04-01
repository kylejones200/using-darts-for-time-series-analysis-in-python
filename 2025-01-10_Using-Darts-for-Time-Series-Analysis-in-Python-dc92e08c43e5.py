# Description: Short example for Using Darts for Time Series Analysis in Python.




# Load the ERCOT data

from darts import TimeSeries
from darts.metrics import mape
from darts.models import ARIMA
from darts.models import ExponentialSmoothing
from data_io import read_csv
import matplotlib.pyplot as plt
import pandas as pd

df = read_csv("ercot_load_data.csv")
df['date'] = pd.to_datetime(df['date'])  # Ensure 'date' is in datetime format
df['values'] = pd.to_numeric(df['values'], errors='coerce')  # Convert 'values' to numeric
df = df.sort_values('date')  # Sort by date

# Drop rows with missing or NaN values
df = df.dropna()

# Resample the data to hourly frequency
df = df.set_index('date').resample('h').mean().reset_index()  # Resample and take the mean for each hour

# Define hold-out period
hold_out_hours = 24  # 24 hours = 1 day
train = df.iloc[:-hold_out_hours]
hold_out = df.iloc[-hold_out_hours:]

# Create TimeSeries for training and hold-out data
series_train = TimeSeries.from_dataframe(train, 'date', 'values', freq="h", fill_missing_dates=True)
series_hold_out = TimeSeries.from_dataframe(hold_out, 'date', 'values', freq="h")

# Fit the Exponential Smoothing model on training data
model = ExponentialSmoothing()
model.fit(series_train)

# Forecast the hold-out period
forecast = model.predict(len(series_hold_out))

# Calculate MAPE
mape = mape(series_hold_out, forecast)

# Plot the results
plt.figure(figsize=(12, 6))

# Plot training data
series_train.plot(label="Training Data", color='blue')

# Plot hold-out data
series_hold_out.plot(label="Hold-Out Data (Actual)", color='green')

# Plot forecasted data
forecast.plot(label="Forecast", color='red')

plt.title(f"ERCOT Hourly Load Forecast with Hold-Out Data \n MAPE: {mape:.2f}%")
plt.xlabel("Date")
plt.ylabel("Load Values")
plt.legend()
plt.tight_layout()
plt.savefig("ERCOT_Hourly_HoldOut_Forecast.png")
plt.show()



# Define hold-out period
hold_out_hours = 24  # Example: 24 hours = 1 day
train = df.iloc[:-hold_out_hours]
hold_out = df.iloc[-hold_out_hours:]

# Create TimeSeries for training and hold-out data
series_train = TimeSeries.from_dataframe(train, 'date', 'values', freq="h", fill_missing_dates=True)
series_hold_out = TimeSeries.from_dataframe(hold_out, 'date', 'values', freq="h")

# Fit the ARIMA model
model = ARIMA(p=1, d=1, q=1)  # You can adjust p, d, q parameters
model.fit(series_train)


forecast = model.predict(len(series_hold_out))
mape_result = mape(series_hold_out, forecast)

plt.figure(figsize=(12, 6))
series_train.plot(label="Training Data", color='blue')
series_hold_out.plot(label="Hold-Out Data (Actual)", color='green')
forecast.plot(label="Forecast", color='red')
plt.title(f"ERCOT Hourly Load Forecast with ARIMA and Hold-Out Period \n MAPE: {mape_result:.2f}%")
plt.xlabel("Date")
plt.ylabel("Load Values")
plt.legend()
plt.tight_layout()
plt.savefig("ARIMA_Hourly_HoldOut_Forecast.png")
plt.show()
