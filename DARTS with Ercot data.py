"""Generated from Jupyter notebook: DARTS with Ercot data

Magics and shell lines are commented out. Run with a normal Python interpreter."""

import matplotlib.pyplot as plt
import pandas as pd
from darts import TimeSeries
from darts.metrics import mape
from darts.models import ExponentialSmoothing


def main():
    df = pd.read_csv("ercot_load_data.csv")
    df["date"] = pd.to_datetime(df["date"])
    df["values"] = pd.to_numeric(df["values"], errors="coerce")
    df = df.sort_values("date")
    df = df.dropna()
    df = df.set_index("date").resample("h").mean().reset_index()
    hold_out_hours = 24
    train = df.iloc[:-hold_out_hours]
    hold_out = df.iloc[-hold_out_hours:]
    series_train = TimeSeries.from_dataframe(
        train, "date", "values", freq="h", fill_missing_dates=True
    )
    series_hold_out = TimeSeries.from_dataframe(hold_out, "date", "values", freq="h")
    model = ExponentialSmoothing()
    model.fit(series_train)
    forecast = model.predict(len(series_hold_out))
    mape_result = mape(series_hold_out, forecast)
    plt.figure(figsize=(12, 6))
    series_train.plot(label="Training Data", color="blue")
    series_hold_out.plot(label="Hold-Out Data (Actual)", color="green")
    forecast.plot(label="Forecast", color="red")
    plt.title(
        f"ERCOT Hourly Load Forecast with Hold-Out Data \n MAPE: {mape_result:.2f}%"
    )
    plt.xlabel("Date")
    plt.ylabel("Load Values")
    plt.legend()
    plt.grid(False)
    plt.tight_layout()
    plt.savefig("ERCOT_Hourly_HoldOut_Forecast.png")
    plt.show()
    "\n    ARIMA\n    "
    import matplotlib.pyplot as plt
    from darts import TimeSeries
    from darts.models import ARIMA

    model = ARIMA(p=1, d=1, q=1)
    model.fit(series)
    forecast = model.predict(n=30, num_samples=1000)
    plt.figure(figsize=(12, 6))
    series[-365:].plot(label="Actual", color="blue")
    forecast.plot(label="Forecast", color="red")
    plt.title(
        "10-Year Treasury Constant Maturity Minus 2-Year Treasury Constant Maturity"
    )
    plt.xlabel("Date")
    plt.ylabel("Spread")
    plt.legend()
    plt.grid(False)
    plt.tight_layout()
    plt.savefig("ARIMA_Forecast.png")
    plt.show()
    import matplotlib.pyplot as plt
    from darts import TimeSeries
    from darts.models import ARIMA

    hold_out_hours = 24
    train = df.iloc[:-hold_out_hours]
    hold_out = df.iloc[-hold_out_hours:]
    series_train = TimeSeries.from_dataframe(
        train, "date", "values", freq="h", fill_missing_dates=True
    )
    series_hold_out = TimeSeries.from_dataframe(hold_out, "date", "values", freq="h")
    model = ARIMA(p=1, d=1, q=1)
    model.fit(series_train)
    forecast = model.predict(len(series_hold_out))
    mape_result = mape(series_hold_out, forecast)
    plt.figure(figsize=(12, 6))
    series_train.plot(label="Training Data", color="blue")
    series_hold_out.plot(label="Hold-Out Data (Actual)", color="green")
    forecast.plot(label="Forecast", color="red")
    plt.title(
        f"ERCOT Hourly Load Forecast with ARIMA and Hold-Out Period \n MAPE: {mape_result:.2f}%"
    )
    plt.xlabel("Date")
    plt.ylabel("Load Values")
    plt.legend()
    plt.grid(False)
    plt.tight_layout()
    plt.savefig("ARIMA_Hourly_HoldOut_Forecast.png")
    plt.show()


def main() -> None:
    main()


if __name__ == "__main__":
    main()
