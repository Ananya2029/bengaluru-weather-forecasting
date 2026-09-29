"""Train and evaluate next-day temperature and rain forecasts for Bengaluru.

Run from the project root:
    python src/train.py
"""
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor  # noqa: E402
from sklearn.linear_model import LogisticRegression, Ridge  # noqa: E402
from sklearn.metrics import (f1_score, mean_absolute_error, precision_score,  # noqa: E402
                             r2_score, recall_score, roc_auc_score, root_mean_squared_error)
from sklearn.model_selection import TimeSeriesSplit, cross_val_score  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from xgboost import XGBClassifier, XGBRegressor  # noqa: E402

from data import load_clean, build_features  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
TEST_START = "2025-04-01"  # last ~12 months held out, so the test covers every season


def split(X, y):
    train = X.index < TEST_START
    return X[train], X[~train], y[train], y[~train]


def temperature_models(X_train, X_test, y_train, y_test, raw):
    y_train, y_test = y_train["temp_avg_tomorrow"], y_test["temp_avg_tomorrow"]
    results, preds = {}, {}

    # Baselines a model has to beat
    preds["Persistence (today = tomorrow)"] = X_test["temp_avg"].to_numpy()
    climatology = raw["temp_avg"][raw.index < TEST_START].groupby(raw.index.dayofyear[raw.index < TEST_START]).mean()
    preds["Climatology (avg for that date)"] = climatology.reindex(
        (X_test.index + pd.Timedelta(days=1)).dayofyear).to_numpy()

    models = {
        "Ridge Regression": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "Random Forest": RandomForestRegressor(n_estimators=400, min_samples_leaf=3, random_state=42, n_jobs=-1),
        "XGBoost": XGBRegressor(n_estimators=400, max_depth=3, learning_rate=0.03,
                                subsample=0.9, colsample_bytree=0.8, random_state=42),
    }
    cv = TimeSeriesSplit(n_splits=5)
    fitted = {}
    for name, model in models.items():
        cv_mae = -cross_val_score(model, X_train, y_train, cv=cv, scoring="neg_mean_absolute_error").mean()
        model.fit(X_train, y_train)
        preds[name] = model.predict(X_test)
        fitted[name] = (model, cv_mae)

    for name, p in preds.items():
        results[name] = {
            "MAE_C": round(mean_absolute_error(y_test, p), 3),
            "RMSE_C": round(root_mean_squared_error(y_test, p), 3),
            "R2": round(r2_score(y_test, p), 3),
            "CV_MAE_C": round(fitted[name][1], 3) if name in fitted else None,
        }

    best = min(fitted, key=lambda n: fitted[n][1])  # choose by CV, not by test score
    return results, preds, best, fitted[best][0], y_test


def rain_models(X_train, X_test, y_train, y_test):
    y_train, y_test = y_train["rain_tomorrow"], y_test["rain_tomorrow"]
    results, probs = {}, {}

    # Baseline: it will rain tomorrow if it rained today
    today = (X_test["precip"] > 0.5).astype(int)
    results["Persistence (rain today -> rain tomorrow)"] = {
        "ROC_AUC": None,
        "Precision": round(precision_score(y_test, today), 3),
        "Recall": round(recall_score(y_test, today), 3),
        "F1": round(f1_score(y_test, today), 3),
    }

    models = {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")),
        "Random Forest": RandomForestClassifier(n_estimators=400, min_samples_leaf=3, class_weight="balanced",
                                                random_state=42, n_jobs=-1),
        "XGBoost": XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.9,
                                 colsample_bytree=0.8, eval_metric="logloss", random_state=42,
                                 scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum()),
    }
    fitted = {}
    for name, model in models.items():
        cv_auc = cross_val_score(model, X_train, y_train, cv=TimeSeriesSplit(n_splits=5), scoring="roc_auc").mean()
        model.fit(X_train, y_train)
        prob = model.predict_proba(X_test)[:, 1]
        pred = (prob >= 0.5).astype(int)
        probs[name] = prob
        fitted[name] = (model, cv_auc)
        results[name] = {
            "ROC_AUC": round(roc_auc_score(y_test, prob), 3),
            "Precision": round(precision_score(y_test, pred), 3),
            "Recall": round(recall_score(y_test, pred), 3),
            "F1": round(f1_score(y_test, pred), 3),
            "CV_ROC_AUC": round(cv_auc, 3),
        }
    best = max(fitted, key=lambda n: fitted[n][1])
    return results, best, fitted[best][0]


def plot_eda(raw):
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    axes[0].fill_between(raw.index, raw["temp_min"], raw["temp_max"], alpha=0.3, label="Min–max")
    axes[0].plot(raw.index, raw["temp_avg"], lw=0.8, label="Average")
    axes[0].set_ylabel("Temperature (°C)")
    axes[0].legend(loc="upper right")
    axes[1].plot(raw.index, raw["humidity_avg"], lw=0.8, color="tab:green")
    axes[1].set_ylabel("Humidity (%)")
    axes[2].bar(raw.index, raw["precip"], width=1.0, color="tab:blue")
    axes[2].set_ylabel("Rain (mm)")
    fig.suptitle("Bengaluru daily weather, 2022–2026")
    fig.tight_layout()
    fig.savefig(OUT / "eda_timeseries.png", dpi=120)
    plt.close(fig)

    monthly = raw.groupby(raw.index.month).agg(temp=("temp_avg", "mean"),
                                               rain_days=("precip", lambda s: (s > 0.5).mean() * 30))
    fig, ax1 = plt.subplots(figsize=(9, 4))
    ax1.bar(monthly.index, monthly["rain_days"], color="tab:blue", alpha=0.6, label="Rain days / month")
    ax1.set_ylabel("Rain days per month")
    ax1.set_xticks(range(1, 13), ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])
    ax2 = ax1.twinx()
    ax2.plot(monthly.index, monthly["temp"], color="tab:red", marker="o", label="Avg temp")
    ax2.set_ylabel("Average temperature (°C)")
    ax1.set_title("Seasonality: hot dry April, monsoon June–October")
    fig.tight_layout()
    fig.savefig(OUT / "eda_seasonality.png", dpi=120)
    plt.close(fig)


def plot_forecast(y_test, preds, best):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(y_test.index, y_test, label="Actual", color="black", lw=1)
    ax.plot(y_test.index, preds[best], label=f"{best} forecast", color="tab:red", lw=1)
    ax.set_ylabel("Next-day avg temperature (°C)")
    ax.set_title("Hold-out year: actual vs forecast")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "forecast_vs_actual.png", dpi=120)
    plt.close(fig)


def plot_importance(model, columns, name):
    est = model[-1] if hasattr(model, "steps") else model
    if not hasattr(est, "feature_importances_"):
        return
    imp = pd.Series(est.feature_importances_, index=columns).sort_values().tail(12)
    fig, ax = plt.subplots(figsize=(8, 5))
    imp.plot.barh(ax=ax, color="tab:blue")
    ax.set_title(f"Top features — {name}")
    fig.tight_layout()
    fig.savefig(OUT / "feature_importance.png", dpi=120)
    plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    raw = load_clean()
    print(f"Loaded {len(raw)} days ({raw.index.min():%d %b %Y} – {raw.index.max():%d %b %Y}), "
          f"{raw.attrs['n_missing_days_filled']} missing day(s) interpolated, "
          f"{raw.attrs['n_sentinel_readings_fixed']} faulty zero readings repaired")
    plot_eda(raw)

    X, y = build_features(raw)
    X_train, X_test, y_train, y_test = split(X, y)
    print(f"Train: {len(X_train)} days | Test: {len(X_test)} days (from {TEST_START})")

    temp_results, temp_preds, temp_best, temp_model, y_temp = temperature_models(X_train, X_test, y_train, y_test, raw)
    rain_results, rain_best, rain_model = rain_models(X_train, X_test, y_train, y_test)

    print("\nNext-day average temperature (test year):")
    print(pd.DataFrame(temp_results).T.to_string())
    print(f"Selected by cross-validation: {temp_best}")
    print("\nRain tomorrow (test year, rain-day rate "
          f"{y_test['rain_tomorrow'].mean():.0%}):")
    print(pd.DataFrame(rain_results).T.to_string())
    print(f"Selected by cross-validation: {rain_best}")

    plot_forecast(y_temp, temp_preds, temp_best)
    plot_importance(temp_model, X.columns, temp_best)

    joblib.dump({"model": temp_model, "features": list(X.columns)}, OUT / "temperature_model.joblib")
    joblib.dump({"model": rain_model, "features": list(X.columns)}, OUT / "rain_model.joblib")
    (OUT / "metrics.json").write_text(json.dumps({
        "temperature": temp_results, "temperature_selected": temp_best,
        "rain": rain_results, "rain_selected": rain_best,
        "test_start": TEST_START, "n_train": len(X_train), "n_test": len(X_test),
    }, indent=2))
    print(f"\nSaved models, metrics and plots to {OUT}")


if __name__ == "__main__":
    main()
