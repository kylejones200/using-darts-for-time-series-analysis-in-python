# Forecasting the U.S. Treasury Yield Spread using Darts

import logging
from datetime import datetime

import pandas as pd
from darts import TimeSeries

logging.basicConfig(level=logging.INFO, format="%(message)s")
logging.getLogger("py.warnings").setLevel(logging.ERROR)




# Fetch and clean data from FRED
def fetch_fred_series(series_id, start="2000-01-01", end=None):
    """Fetch FRED data as a Darts TimeSeries via pandas_datareader."""
    import pandas_datareader.data as web

    if end is None:
        end = datetime.now().strftime("%Y-%m-%d")
    df = web.DataReader(series_id, "fred", start=start, end=end)
    df = df.rename(columns={series_id: "value"})
    df["value"] = pd.to_numeric(df["value"], errors="coerce").ffill().dropna()
    return TimeSeries.from_dataframe(df.sort_index(), value_cols="value")
