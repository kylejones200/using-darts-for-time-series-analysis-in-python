"""Generated from Jupyter notebook: DARTS with NBEATS on Ercot data

Magics and shell lines are commented out. Run with a normal Python interpreter."""

import matplotlib.pyplot as plt
import pandas as pd
from darts import TimeSeries
from darts.metrics import mae, mape, rmse
from darts.models import NBEATSModel


def load_and_preprocess_the_ercot_data() -> None:
    df = pd.read_csv("ercot_load_data.csv")
    df["Date"] = pd.to_datetime(df["date"])
    df["Value"] = pd.to_numeric(df["values"], errors="coerce")
    df = df.sort_values("Date")
    df = df.dropna(subset=["Date", "Value"])
    df = df.set_index("Date").resample("H")["Value"].mean().reset_index()
    series = TimeSeries.from_dataframe(df, time_col="Date", value_cols="Value")
    train, test = series.split_before(0.8)
    model = NBEATSModel(
        input_chunk_length=30, output_chunk_length=10, n_epochs=50, random_state=42
    )
    model.fit(train)
    backtest_results = model.backtest(
        series,
        start=0.8,
        forecast_horizon=10,
        stride=1,
        retrain=False,
        verbose=True,
        metric=[mape, rmse, mae],
    )
    print("Backtest Results:")
    print(f"MAPE: {backtest_results[0]:.2f}%")
    print(f"RMSE: {backtest_results[1]:.2f}")
    print(f"MAE: {backtest_results[2]:.2f}")
    historical_forecasts = model.historical_forecasts(
        series, start=0.8, forecast_horizon=10, stride=1, retrain=False, verbose=True
    )
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


def notebook_step_002() -> None:
    df.head()


def load_and_preprocess_the_ercot_data_2() -> None:
    df = pd.read_csv("ercot_load_data.csv")
    df["Date"] = pd.to_datetime(df["date"])
    df["Value"] = pd.to_numeric(df["values"], errors="coerce")
    df = df.sort_values("Date")
    df = df.dropna(subset=["Date", "Value"])
    df = df.set_index("Date").resample("h")["Value"].mean().reset_index()
    df = df.tail(100)
    series = TimeSeries.from_dataframe(df, time_col="Date", value_cols="Value")
    train, test = series.split_before(0.8)
    model = NBEATSModel(
        input_chunk_length=12,
        output_chunk_length=6,
        n_epochs=5,
        batch_size=8,
        random_state=42,
        force_reset=True,
    )
    try:
        model.fit(train)
    except RuntimeError as e:
        print("Model fitting failed:", e)

    try:
        backtest_results = model.backtest(
            series,
            start=0.8,
            forecast_horizon=12,
            stride=1,
            retrain=False,
            verbose=True,
            metric=[mape, rmse, mae],
        )
        print("Backtest Results:")
        print(f"MAPE: {backtest_results[0]:.2f}%")
        print(f"RMSE: {backtest_results[1]:.2f}")
        print(f"MAE: {backtest_results[2]:.2f}")
    except RuntimeError as e:
        print("Backtesting failed:", e)

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


def main() -> None:
    load_and_preprocess_the_ercot_data()
    notebook_step_002()
    load_and_preprocess_the_ercot_data_2()


if __name__ == "__main__":
    main()
