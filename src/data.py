"""Load and clean the Bengaluru daily weather data, and build forecasting features."""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "Bengaluru_Weather.csv"

COLUMNS = [
    "month", "day", "year",
    "temp_max", "temp_avg", "temp_min",
    "dew_max", "dew_avg", "dew_min",
    "humidity_max", "humidity_avg", "humidity_min",
    "wind_max", "wind_avg", "wind_min",
    "pressure_max", "pressure_avg", "pressure_min",
    "precip",
]
WEATHER_VARS = COLUMNS[3:]


def f_to_c(f):
    return (f - 32) * 5 / 9


def load_clean(path=RAW):
    """Return one row per day, indexed by date, temperatures in °C and rain in mm."""
    df = pd.read_csv(path, header=None, skiprows=2, names=COLUMNS, encoding="latin1")
    df = df.dropna(how="all")  # trailing blank rows in the export

    df["date"] = pd.to_datetime(
        df["year"].astype(int).astype(str) + "-" + df["month"] + "-" + df["day"].astype(int).astype(str),
        format="%Y-%b-%d",
    )
    df = df.set_index("date").drop(columns=["month", "day", "year"]).sort_index()

    # The station logs a failed reading as 0 (0 °F, 0 % humidity, 0 inHg), which is
    # physically impossible in Bengaluru. Such a zero also drags that day's average down,
    # so treat both the minimum and the average as missing and interpolate them below.
    n_sentinel = 0
    for var in ("temp", "dew", "humidity", "pressure"):
        bad = df[f"{var}_min"] == 0
        n_sentinel += int(bad.sum())
        df.loc[bad, [f"{var}_min", f"{var}_avg"]] = np.nan

    for col in ["temp_max", "temp_avg", "temp_min", "dew_max", "dew_avg", "dew_min"]:
        df[col] = f_to_c(df[col])
    df["precip"] = df["precip"] * 25.4  # inches -> mm

    # Make the calendar continuous; fill the few missing days by time interpolation
    full_range = pd.date_range(df.index.min(), df.index.max(), freq="D")
    n_missing = len(full_range) - len(df)
    df = df.reindex(full_range).interpolate(method="time")
    df.index.name = "date"
    df.attrs["n_missing_days_filled"] = n_missing
    df.attrs["n_sentinel_readings_fixed"] = n_sentinel
    return df


def build_features(df):
    """Features known at the end of day t, used to predict day t+1."""
    feats = df[WEATHER_VARS].copy()

    for lag in (1, 2, 3, 7):
        feats[f"temp_avg_lag{lag}"] = df["temp_avg"].shift(lag)
    feats["humidity_avg_lag1"] = df["humidity_avg"].shift(1)
    feats["precip_lag1"] = df["precip"].shift(1)

    for window in (3, 7, 30):
        feats[f"temp_avg_roll{window}"] = df["temp_avg"].rolling(window).mean()
    feats["precip_roll7"] = df["precip"].rolling(7).sum()
    feats["rain_days_roll7"] = (df["precip"] > 0).rolling(7).sum()
    feats["temp_range"] = df["temp_max"] - df["temp_min"]
    feats["pressure_change"] = df["pressure_avg"].diff()

    # Seasonality encoded on a circle so 31 Dec and 1 Jan are neighbours
    doy = df.index.dayofyear
    feats["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    feats["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)

    targets = pd.DataFrame({
        "temp_avg_tomorrow": df["temp_avg"].shift(-1),
        "rain_tomorrow": (df["precip"].shift(-1) > 0.5).astype(float),  # > 0.5 mm counts as a rain day
    }, index=df.index)
    targets.loc[targets.index[-1], "rain_tomorrow"] = np.nan

    data = feats.join(targets).dropna()
    return data.drop(columns=targets.columns), data[targets.columns]
