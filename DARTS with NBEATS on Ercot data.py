"""Generated from Jupyter notebook: DARTS with NBEATS on Ercot data

Magics and shell lines are commented out. Run with a normal Python interpreter."""


# --- code cell ---

import matplotlib.pyplot as plt
import pandas as pd
from darts import TimeSeries
from darts.metrics import mae, mape, rmse
from darts.models import NBEATSModel


def main():
    # Load and preprocess the ERCOT data
    df = pd.read_csv("ercot_load_data.csv")
    df["Date"] = pd.to_datetime(df["date"])  # Ensure 'date' is in datetime format
    df["Value"] = pd.to_numeric(
        df["values"], errors="coerce"
    )  # Convert 'values' to numeric
    df = df.sort_values("Date")  # Sort by date

    # Drop rows with missing or NaN values
    df = df.dropna(subset=["Date", "Value"])

    # Resample the data to hourly frequency
    df = (
        df.set_index("Date").resample("H")["Value"].mean().reset_index()
    )  # Resample and take the mean for each hour

    # Create a TimeSeries object
    series = TimeSeries.from_dataframe(df, time_col="Date", value_cols="Value")

    # Split the data into train and test sets
    train, test = series.split_before(0.8)

    # Initialize the N-BEATS model
    model = NBEATSModel(
        input_chunk_length=30, output_chunk_length=10, n_epochs=50, random_state=42
    )

    # Fit the model
    model.fit(train)

    # Perform backtesting
    backtest_results = model.backtest(
        series,
        start=0.8,  # Use 80% of the data for the first training
        forecast_horizon=10,
        stride=1,
        retrain=False,  # We've already fitted the model, so we don't need to retrain
        verbose=True,
        metric=[mape, rmse, mae],  # Using multiple metrics
    )

    # Print the backtest results
    print("Backtest Results:")
    print(f"MAPE: {backtest_results[0]:.2f}%")
    print(f"RMSE: {backtest_results[1]:.2f}")
    print(f"MAE: {backtest_results[2]:.2f}")

    # Generate historical forecasts for plotting
    historical_forecasts = model.historical_forecasts(
        series,
        start=0.8,
        forecast_horizon=10,
        stride=1,
        retrain=False,
        verbose=True,
    )

    # Plot the results
    plt.figure(figsize=(12, 6))
    series.plot(label="Actual")
    historical_forecasts.plot(label="Forecast")
    plt.title("N-BEATS - Historical Forecasts")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(False)
    plt.savefig("NBEATS_Backtest.png")
    plt.show()


    # --- code cell ---

    df.head()


    # --- code cell ---

    import matplotlib.pyplot as plt
    import pandas as pd
    from darts import TimeSeries
    from darts.metrics import mae, mape, rmse
    from darts.models import NBEATSModel

    # Load and preprocess the ERCOT data
    df = pd.read_csv("ercot_load_data.csv")
    df["Date"] = pd.to_datetime(df["date"])  # Ensure 'date' is in datetime format
    df["Value"] = pd.to_numeric(
        df["values"], errors="coerce"
    )  # Convert 'values' to numeric
    df = df.sort_values("Date")  # Sort by date
    df = df.dropna(subset=["Date", "Value"])  # Drop rows with missing or NaN values

    # Resample the data to hourly frequency (limit dataset size if necessary)
    df = (
        df.set_index("Date").resample("h")["Value"].mean().reset_index()
    )  # Resample to hourly intervals
    df = df.tail(100)  # Optional: Limit data to the last 1000 rows for testing

    # Create a TimeSeries object
    series = TimeSeries.from_dataframe(df, time_col="Date", value_cols="Value")

    # Split the data into train and test sets
    train, test = series.split_before(0.8)

    # Initialize the N-BEATS model with optimized settings
    model = NBEATSModel(
        input_chunk_length=12,
        output_chunk_length=6,
        n_epochs=5,
        batch_size=8,
        random_state=42,
        force_reset=True,
    )


    # Fit the model
    try:
        model.fit(train)
    except RuntimeError as e:
        print("Model fitting failed:", e)

    # Perform backtesting
    try:
        backtest_results = model.backtest(
            series,
            start=0.8,  # Use 80% of the data for the first training
            forecast_horizon=12,
            stride=1,
            retrain=False,  # We've already fitted the model, so we don't need to retrain
            verbose=True,
            metric=[mape, rmse, mae],  # Using multiple metrics
        )
        print("Backtest Results:")
        print(f"MAPE: {backtest_results[0]:.2f}%")
        print(f"RMSE: {backtest_results[1]:.2f}")
        print(f"MAE: {backtest_results[2]:.2f}")
    except RuntimeError as e:
        print("Backtesting failed:", e)

    # Generate historical forecasts for plotting
    try:
        historical_forecasts = model.historical_forecasts(
            series,
            start=0.8,
            forecast_horizon=12,
            stride=1,
            retrain=False,
            verbose=True,
        )
    except RuntimeError as e:
        print("Historical forecasting failed:", e)

    # Plot the results
    plt.figure(figsize=(12, 6))
    series.plot(label="Actual")
    if "historical_forecasts" in locals():
        historical_forecasts.plot(label="Forecast")
    plt.title("N-BEATS - Historical Forecasts")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(False)
    plt.tight_layout()
    plt.savefig("NBEATS_Backtest_Optimized.png")
    plt.show()


if __name__ == "__main__":
    main()
