"""Generated from Jupyter notebook: sunspots with darts

Magics and shell lines are commented out. Run with a normal Python interpreter."""

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.metrics import mape
from darts.models import (
    TBATS,
    ExponentialSmoothing,
    LinearRegressionModel,
    NHiTSModel,
    RandomForest,
    RNNModel,
    Theta,
)
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from darts.utils.utils import ModelMode, SeasonalityMode
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def animate(frame):
    current_idx = frame + 1
    for name in models.keys():
        pred = results[name]["prediction"]
        lines[name].set_data(pred.time_index[:current_idx], pred.values()[:current_idx])
    return list(lines.values())


def plot_light_curve(data, index):
    flux_values = data.iloc[index, 1:]
    plt.figure(figsize=(15, 5))
    plt.plot(range(len(flux_values)), flux_values)
    plt.title(f"Light Curve (Label: {data.iloc[index, 0]})")
    plt.xlabel("Time Point")
    plt.ylabel("Flux")
    plt.grid(True)
    plt.show()


def prepare_data(data):
    X = data.drop("LABEL", axis=1)
    y = data["LABEL"]
    return (X, y)


def main() -> None:
    df = pd.read_csv("monthly-sunspots.csv")

    df["Sunspots"] = np.where(df["Sunspots"] == 0, 1, df["Sunspots"])

    df = df[-800:].reset_index(drop=True)

    series = TimeSeries.from_dataframe(df, "Month", "Sunspots")

    train, val = series.split_before(pd.Timestamp("19800101"))

    print("training set: ", len(train))

    print("validation set: ", len(val))

    transformer = Scaler()

    train_transformed = transformer.fit_transform(train)

    val_transformed = transformer.transform(val)

    year_series = datetime_attribute_timeseries(
        pd.date_range(
            start=series.start_time(), freq=series.freq_str, end=series.end_time()
        ),
        attribute="year",
        one_hot=False,
    )

    covariates = Scaler().fit_transform(year_series)

    models = {
        "Exponential Smoothing": ExponentialSmoothing(
            trend=ModelMode.ADDITIVE,
            seasonal=SeasonalityMode.ADDITIVE,
            seasonal_periods=124,
        ),
        "Theta": Theta(seasonality_period=124, season_mode=SeasonalityMode.ADDITIVE),
        "Linear Regression": LinearRegressionModel(lags=124, output_chunk_length=20),
        "Random Forest": RandomForest(
            lags=[-1, -12, -124],
            output_chunk_length=20,
            n_estimators=200,
            max_depth=20,
            min_samples_split=10,
            criterion="absolute_error",
        ),
        "RNN": RNNModel(
            model="LSTM",
            training_length=240,
            input_chunk_length=120,
            n_epochs=200,
            batch_size=20,
            optimizer_kwargs={"lr": 0.001},
        ),
        "TBATS": TBATS(
            seasonal_periods=[12],
            use_box_cox=True,
            use_trend=True,
            use_arma_errors=True,
        ),
        "NHiTS": NHiTSModel(
            input_chunk_length=100, output_chunk_length=20, n_epochs=80
        ),
    }

    results = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")
        if name == "RNN":
            model.fit(train_transformed, verbose=True)
            pred = model.predict(len(val))
            pred = transformer.inverse_transform(pred)
        else:
            model.fit(train)
            pred = model.predict(len(val))
        mape_score = np.round(mape(pred, val), 2)
        results[name] = {"prediction": pred, "mape": mape_score}

    fig = plt.figure(figsize=(15, 7))

    ax = plt.axes()

    ax.set_xlim(series.time_index.min(), series.time_index.max())

    ax.set_ylim(0, series.values().max() * 1.1)

    series.plot(label="Actual", ax=ax, color="black", alpha=0.6)

    colors = plt.cm.rainbow(np.linspace(0, 1, len(models)))

    lines = {}

    for name, color in zip(models.keys(), colors):
        (line,) = ax.plot(
            [], [], label=f"{name} (MAPE: {results[name]['mape']}%)", lw=2, color=color
        )
        lines[name] = line

    plt.title("Sunspots Forecast - All Model Predictions", pad=20)

    plt.xlabel("Time")

    plt.ylabel("Sunspots")

    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    anim = animation.FuncAnimation(
        fig, animate, frames=len(models), interval=1000, blit=True, repeat=True
    )

    anim.save("model_predictions.gif", writer="pillow", fps=1, dpi=100)

    plt.close()

    print("\nFinal MAPE Scores:")

    for name, result in results.items():
        print(f"{name}: {result['mape']}%")

    fig = plt.figure(figsize=(15, 7))

    ax = plt.axes()

    start_date = pd.Timestamp("19450101")

    ax.set_xlim(start_date, val.time_index.max())

    ax.set_ylim(0, series.values().max() * 1.1)

    all_times = series.time_index.values

    all_values = series.values().flatten()

    historical_mask = (all_times >= start_date) & (all_times < val.time_index[0])

    (historical_line,) = ax.plot(
        all_times[historical_mask],
        all_values[historical_mask],
        label="Historical",
        color="gray",
        alpha=0.6,
    )

    colors = plt.cm.rainbow(np.linspace(0, 1, len(models)))

    lines = {}

    for name, color in zip(models.keys(), colors):
        (line,) = ax.plot(
            [], [], label=f"{name} (MAPE: {results[name]['mape']}%)", lw=2, color=color
        )
        lines[name] = line

    (val_line,) = ax.plot([], [], label="Actual", color="black", alpha=0.6)

    plt.title("Sunspots Forecast - All Model Predictions", pad=20)

    plt.xlabel("Time")

    plt.ylabel("Sunspots")

    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    anim = animation.FuncAnimation(
        fig, animate, frames=len(val), interval=50, blit=True, repeat=True
    )

    anim.save("model_predictions.gif", writer="pillow", fps=20, dpi=100)

    plt.close()

    print("\nFinal MAPE Scores:")

    for name, result in results.items():
        print(f"{name}: {result['mape']}%")

    train_data = pd.read_csv("Exoplanets-Training-Data.csv")

    test_data = pd.read_csv("Exoplanets-Batch-Test.csv")

    X_train = train_data.drop(["Has Planet"], axis=1)

    y_train = train_data["Has Planet"]

    X_test = test_data.drop(["Has Planet (real)"], axis=1)

    y_test = test_data["Has Planet (real)"]

    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

    rf_classifier.fit(X_train, y_train)

    y_pred = rf_classifier.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    print(f"Model Accuracy: {accuracy:.4f}")

    print("\nClassification Report:")

    print(classification_report(y_test, y_pred))

    feature_importance = pd.DataFrame(
        {"feature": X_train.columns, "importance": rf_classifier.feature_importances_}
    )

    feature_importance = feature_importance.sort_values("importance", ascending=False)

    plt.figure(figsize=(12, 6))

    plt.bar(range(len(feature_importance)), feature_importance["importance"])

    plt.xticks(
        range(len(feature_importance)), feature_importance["feature"], rotation=45
    )

    plt.title("Feature Importance in Planet Detection")

    plt.xlabel("Features")

    plt.ylabel("Importance")

    plt.tight_layout()

    plt.show()

    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(8, 6))

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

    plt.title("Confusion Matrix")

    plt.ylabel("True Label")

    plt.xlabel("Predicted Label")

    plt.show()

    data = pd.read_csv("exoTrain.csv")

    X, y = prepare_data(data)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)

    X_test_scaled = scaler.transform(X_test)

    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

    rf_classifier.fit(X_train_scaled, y_train)

    y_pred = rf_classifier.predict(X_test_scaled)

    print("Model Performance:")

    print("-----------------")

    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

    print("\nClassification Report:")

    print(classification_report(y_test, y_pred))

    plt.figure(figsize=(8, 6))

    cm = confusion_matrix(y_test, y_pred)

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

    plt.title("Confusion Matrix")

    plt.ylabel("True Label")

    plt.xlabel("Predicted Label")

    plt.show()

    feature_importance = pd.DataFrame(
        {"feature": X.columns, "importance": rf_classifier.feature_importances_}
    )

    feature_importance = feature_importance.sort_values("importance", ascending=False)

    plt.figure(figsize=(12, 6))

    plt.bar(range(20), feature_importance["importance"][:20])

    plt.xticks(range(20), feature_importance["feature"][:20], rotation=45)

    plt.title("Top 20 Most Important Features in Planet Detection")

    plt.xlabel("Features")

    plt.ylabel("Importance")

    plt.tight_layout()

    plt.show()

    for label in [1, 2]:
        sample_index = data[data["LABEL"] == label].index[0]
        plot_light_curve(data, sample_index)

    df = pd.read_csv("monthly-sunspots.csv")

    df["Sunspots"] = np.where(df["Sunspots"] == 0, 1, df["Sunspots"])

    df = df[-800:].reset_index(drop=True)

    df["Month"] = pd.to_datetime(df["Month"])

    df.set_index("Month", inplace=True)

    df_yearly = df.resample("Y").mean()

    series = TimeSeries.from_dataframe(df_yearly, value_cols="Sunspots")

    train, val = series.split_before(pd.Timestamp("19800101"))

    print("training set: ", len(train))

    print("validation set: ", len(val))

    models = {
        "Exponential Smoothing": ExponentialSmoothing(
            trend=ModelMode.ADDITIVE,
            seasonal=SeasonalityMode.ADDITIVE,
            seasonal_periods=11,
        ),
        "Theta": Theta(seasonality_period=11, season_mode=SeasonalityMode.ADDITIVE),
        "Linear Regression": LinearRegressionModel(lags=11, output_chunk_length=20),
        "Random Forest": RandomForest(
            lags=[-1, -2, -11],
            output_chunk_length=20,
            n_estimators=200,
            max_depth=20,
            min_samples_split=10,
            criterion="absolute_error",
        ),
        "TBATS": TBATS(
            seasonal_periods=[11],
            use_box_cox=True,
            use_trend=True,
            use_arma_errors=True,
        ),
        "NHiTS": NHiTSModel(input_chunk_length=10, output_chunk_length=20, n_epochs=80),
    }

    results = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(train)
        pred = model.predict(20)
        results[name] = {"prediction": pred}

    fig = plt.figure(figsize=(15, 7))

    ax = plt.axes()

    start_date = pd.Timestamp("19450101")

    end_date = results[list(models.keys())[0]]["prediction"].time_index[-1]

    historical_df = series.pd_dataframe()

    historical_df = historical_df[historical_df.index >= start_date]

    ax.plot(
        historical_df.index,
        historical_df.values,
        label="Historical",
        color="black",
        alpha=0.6,
    )

    ax.set_xlim(start_date, end_date)

    ax.set_ylim(0, series.values().max() * 1.1)

    colors = plt.cm.rainbow(np.linspace(0, 1, len(models)))

    lines = {}

    for name, color in zip(models.keys(), colors):
        (line,) = ax.plot([], [], label=f"{name}", lw=2, color=color)
        lines[name] = line

    plt.title("Yearly Sunspots Forecast - 20 Years into Future", pad=20)

    plt.xlabel("Time")

    plt.ylabel("Sunspots")

    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    anim = animation.FuncAnimation(
        fig, animate, frames=20, interval=200, blit=True, repeat=True
    )

    anim.save("sunspots_forecast.gif", writer="pillow", fps=5, dpi=100)

    plt.close()

    print("\nFinal predicted values (20 years into future):")

    for name, result in results.items():
        final_value = result["prediction"].values()[-1][0]
        print(f"{name}: {final_value:.2f}")

    df = pd.read_csv("monthly-sunspots.csv")

    df["Sunspots"] = np.where(df["Sunspots"] == 0, 1, df["Sunspots"])

    df = df[-800:].reset_index(drop=True)

    df["Month"] = pd.to_datetime(df["Month"])

    df.set_index("Month", inplace=True)

    df_yearly = df.resample("Y").mean()

    series = TimeSeries.from_dataframe(df_yearly, value_cols="Sunspots")

    train, val = series.split_before(pd.Timestamp("19800101"))

    print("training set: ", len(train))

    print("validation set: ", len(val))

    models = {
        "Exponential Smoothing": ExponentialSmoothing(
            trend=ModelMode.ADDITIVE,
            seasonal=SeasonalityMode.ADDITIVE,
            seasonal_periods=11,
        ),
        "Theta": Theta(seasonality_period=11, season_mode=SeasonalityMode.ADDITIVE),
        "Linear Regression": LinearRegressionModel(lags=11, output_chunk_length=20),
        "Random Forest": RandomForest(
            lags=[-1, -2, -11],
            output_chunk_length=20,
            n_estimators=200,
            max_depth=20,
            min_samples_split=10,
            criterion="absolute_error",
        ),
        "TBATS": TBATS(
            seasonal_periods=[11],
            use_box_cox=True,
            use_trend=True,
            use_arma_errors=True,
        ),
        "NHiTS": NHiTSModel(input_chunk_length=10, output_chunk_length=20, n_epochs=80),
    }

    results = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(train)
        pred = model.predict(20)
        results[name] = {"prediction": pred}

    plt.figure(figsize=(15, 8))

    start_date = pd.Timestamp("19450101")

    historical_df = series.pd_dataframe()

    historical_df = historical_df[historical_df.index >= start_date]

    plt.plot(
        historical_df.index,
        historical_df.values,
        label="Historical",
        color="black",
        alpha=0.6,
        linewidth=2,
    )

    colors = plt.cm.rainbow(np.linspace(0, 1, len(models)))

    for name, color in zip(models.keys(), colors):
        pred = results[name]["prediction"]
        plt.plot(
            pred.time_index, pred.values(), label=f"{name}", color=color, linewidth=2
        )

    prediction_start = historical_df.index[-1]

    plt.axvline(x=prediction_start, color="gray", linestyle="--", alpha=0.5)

    plt.text(
        prediction_start,
        plt.ylim()[1],
        "Prediction Start",
        rotation=90,
        verticalalignment="top",
    )

    plt.title("Sunspots: Historical Data and 20-Year Forecasts", pad=20, size=14)

    plt.xlabel("Year", size=12)

    plt.ylabel("Sunspots", size=12)

    plt.grid(True, alpha=0.3)

    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()

    plt.savefig("sunspots_all_predictions.png", dpi=300, bbox_inches="tight")

    plt.show()

    print("\nFinal predicted values (20 years into future):")

    for name, result in results.items():
        final_value = result["prediction"].values()[-1][0]
        print(f"{name}: {final_value:.2f}")

    end_date = pd.Timestamp("20260101")

    train_end = train.time_index[-1]

    years_to_predict = end_date.year - train_end.year + 1

    results = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(train)
        pred = model.predict(years_to_predict)
        results[name] = {"prediction": pred}

    target_date = pd.Timestamp("20250101")

    predictions_2025 = {}

    for name, result in results.items():
        pred_df = result["prediction"].pd_dataframe()
        value_2025 = pred_df.loc[target_date]["Sunspots"]
        predictions_2025[name] = value_2025

    print("\nPredictions for January 2025:")

    total = 0

    for name, value in predictions_2025.items():
        print(f"{name}: {value:.2f}")
        total += value

    mean_prediction = total / len(predictions_2025)

    print(f"\nMean prediction for January 2025: {mean_prediction:.2f}")

    end_date = pd.Timestamp("20260101")

    train_end = train.time_index[-1]

    years_to_predict = end_date.year - train_end.year + 1

    results = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(train)
        pred = model.predict(years_to_predict)
        results[name] = {"prediction": pred}

    predictions_2025 = {}

    for name, result in results.items():
        pred_df = result["prediction"].pd_dataframe()
        value_2025 = pred_df[pred_df.index.year == 2025].iloc[0]["Sunspots"]
        predictions_2025[name] = value_2025

    print("\nPredictions for 2025:")

    total = 0

    for name, value in predictions_2025.items():
        print(f"{name}: {value:.2f}")
        total += value

    mean_prediction = total / len(predictions_2025)

    print(f"\nMean prediction for 2025: {mean_prediction:.2f}")

    plt.figure(figsize=(15, 8))

    start_date = pd.Timestamp("19450101")

    historical_df = series.pd_dataframe()

    historical_df = historical_df[historical_df.index >= start_date]

    plt.plot(
        historical_df.index,
        historical_df.values,
        label="Historical",
        color="black",
        alpha=0.6,
        linewidth=2,
    )

    colors = plt.cm.rainbow(np.linspace(0, 1, len(models)))

    for name, color in zip(models.keys(), colors):
        pred = results[name]["prediction"]
        plt.plot(
            pred.time_index, pred.values(), label=f"{name}", color=color, linewidth=2
        )

    prediction_start = historical_df.index[-1]

    plt.axvline(x=prediction_start, color="gray", linestyle="--", alpha=0.5)

    plt.text(
        prediction_start,
        plt.ylim()[1],
        "Prediction Start",
        rotation=90,
        verticalalignment="top",
    )

    plt.title("Sunspots: Historical Data and Forecasts to 2026", pad=20, size=14)

    plt.xlabel("Year", size=12)

    plt.ylabel("Sunspots", size=12)

    plt.grid(True, alpha=0.3)

    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()

    plt.savefig("sunspots_all_predictions.png", dpi=300, bbox_inches="tight")

    plt.show()


if __name__ == "__main__":
    main()
