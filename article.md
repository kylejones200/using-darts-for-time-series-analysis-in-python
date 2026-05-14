# Using Darts for Time Series Analysis in Python

Darts simplifies time series analysis by providing a unified interface for multiple forecasting methods, from traditional statistical...

### Using Darts for Time Series Analysis in Python 

#### Darts simplifies time series analysis by providing a unified interface for multiple forecasting methods, from traditional statistical approaches to advanced machine learning models.
Darts is a time series and forecasting library that streamlines building, evaluating, and deploying time series models. It is basically a wrapper for a lot of other models from ARIMA to LSTM++.

Darts provides a unified API for a variety of time series models so we can switch between models and compare performance. It has built in preprocessing for resampling, scaling, and handling missing data. And it comes with evaluation tools for standard metrics like MAE, MAPE, RMSE, and cross-validation.

Let's try it out.

Install Darts with: `pip install darts`

#### Basic Workflow
The typical workflow in Darts involves creating a `TimeSeries` object, choosing and training a model, making predictions, and evaluating results.

Darts requires data to be in a `TimeSeries` object, which wraps around Pandas DataFrames.

In this project I use data from FRED (Federal Reserve Bank of St. Louis) on 10-Year Treasury Constant Maturity Minus 2-Year Treasury Constant Maturity (T10Y2Y). The T10Y2Y is a financial indicator that measures the difference between the yields of 10-year and 2-year U.S. Treasury securities (aka the "yield spread" or the "term spread") in basis points. Positive spread means that 10-year bonds have higher yields than 2-year bonds, which is the normal situation. Negative speard (aka yield curve inversion) is a sign of potential economic slowdown or recession.

Economists and investors use this as an indicator of market expectations about future economic conditions and monetary policy.

FRED provides this data by API which is easy to use with pandas_datareader.


Darts supports traditional methods like Exponential Smoothing and ARIMA for quick, interpretable forecasts.

#### Exponential Smoothing


I included way too much history in this version. The forecast is just 10 steps so it is a little blue dot at the end that you can barely see.

#### ARIMA
The autoARIMA implementations in other tools like statsmodels are easier to use than Darts. But let's still look at it.


In this viz, I zoomed so it is much easier to see the prediction --- which is basically a straight line for the next 30 days.

### Machine Learning Models in Darts
But that is just the start. Darts can also do more advanced forecasting with machine learning models like Random Forests and LightGBM.

I have another article that goes deeper into N-BEATS.

[N-BEATS for Time Series Forecasting in Python\ *N-BEATS (Neural Basis Expansion Analysis for Time Series) is a deep learning model specifically designed for time...*medium.com](https://medium.com/@kylejones_47003/n-beats-for-time-series-forecasting-in-python-b4a61858fe49 "https://medium.com/@kylejones_47003/n-beats-for-time-series-forecasting-in-python-b4a61858fe49")[](https://medium.com/@kylejones_47003/n-beats-for-time-series-forecasting-in-python-b4a61858fe49)


### Evaluating Models
Darts makes it easy to evaluate model performance using metrics like MAE, RMSE, and MAPE.


#### Real world example: Energy Load Data
I pulled data for energy load in ERCOT for every 15 mins from Dec 24, 2024 to January 11, 2025. Then I repeated these steps using the new dataset.



The exponential smoothing works better than ARIMA on this dataset.



### But wait, there's more!
Darts has tools for Backtesting to Evaluate how well a model performs over historical data. It can do Transformations like Scale, log-transform, or normalize data before modeling. And it can do ensembling to combine multiple models for better forecasts.


### Deployment with Darts
You can pickle Darts models so they are easy to containerize for inference.


I got a little carried away and made this version that uses N-BEATS and Fast Fourier Transforms. The FFTs do a terrible job on this dataset but it was still fun to make.


### Key Takeaways
Darts is a pretty good time series and forecasting library with an intuitive API and a lot of options. It covers most of the things you need for day to day TS work.
