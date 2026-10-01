"""
EmergeRoute — Traffic Prediction Module (XGBoost)

Predicts near-future congestion (15-30 minutes ahead, per the brief) from
recent traffic history, using XGBoost regression on lag features. This lets
EmergeRoute act proactively -- feeding a PREDICTED congestion level into the
policy generator, rather than only reacting to congestion that has already
happened.

Trained on time-series data produced by log_traffic_data.py (from our own
SUMO simulation -- see that file's docstring for why). The same feature
engineering / model would work unchanged on a real dataset like METR-LA if
swapped in later; only the CSV loading changes.

Target: n_halting (queued/stopped vehicles) at time T, predicted from lag
features at times T-1, T-2, T-3 (i.e. the last 3 logged intervals).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


N_LAGS = 3  # how many past time-steps to use as predictive features


def build_features(df: pd.DataFrame, n_lags: int = N_LAGS) -> tuple[pd.DataFrame, pd.Series]:
    """Turns a raw time-series into a supervised-learning table: each row's
    features are the previous n_lags steps' (n_vehicles, avg_speed, n_halting),
    and the target is the CURRENT step's n_halting (what we want to predict
    ahead of time)."""
    df = df.sort_values("time_s").reset_index(drop=True)
    feature_cols = ["n_vehicles", "avg_speed", "n_halting"]

    rows = []
    targets = []
    for i in range(n_lags, len(df)):
        feat_row = {}
        for lag in range(1, n_lags + 1):
            past = df.iloc[i - lag]
            for col in feature_cols:
                feat_row[f"{col}_lag{lag}"] = past[col]
        rows.append(feat_row)
        targets.append(df.iloc[i]["n_halting"])

    X = pd.DataFrame(rows)
    y = pd.Series(targets, name="n_halting_target")
    return X, y


def train_predictor(csv_path: str = "traffic_log.csv", n_lags: int = N_LAGS):
    df = pd.read_csv(csv_path)
    X, y = build_features(df, n_lags=n_lags)

    if len(X) < 10:
        raise ValueError(
            f"Only {len(X)} training rows available from {csv_path} -- log a "
            "longer/finer-grained simulation run (see log_traffic_data.py) "
            "for a meaningful train/test split."
        )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
    # shuffle=False: keep chronological order for train/test split, since
    # this is time-series data -- testing on "future" data relative to
    # training is the honest evaluation here, not a random shuffle.

    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    print(f"Trained on {len(X_train)} samples, tested on {len(X_test)} samples.")
    print(f"Mean Absolute Error: {mae:.2f} vehicles (queue length prediction)")
    print(f"Test set actual range: {y_test.min():.0f}-{y_test.max():.0f} vehicles")

    return model, X.columns.tolist()


def predict_congestion_level(model, feature_columns: list[str], recent_history: pd.DataFrame) -> str:
    """Given the most recent n_lags rows of real traffic data, predicts the
    NEXT interval's queue length and classifies it into low/moderate/high --
    directly consumable by policy_generator.generate_candidate_policies().
    """
    X, _ = build_features(pd.concat([recent_history, recent_history.iloc[[-1]]]), n_lags=N_LAGS)
    # ^ append a dummy final row so build_features has something to predict
    #   FOR (its target value is discarded, only the lag features matter)
    if len(X) == 0:
        return "moderate"  # not enough history yet, default to a cautious middle ground

    predicted_halting = model.predict(X.iloc[[-1]])[0]

    if predicted_halting < 20:
        return "low"
    elif predicted_halting < 80:
        return "moderate"
    else:
        return "high"


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="traffic_log.csv")
    args = parser.parse_args()

    model, feature_columns = train_predictor(args.csv)

    # Demo: use the last few rows of real logged data to predict what comes next
    df = pd.read_csv(args.csv).sort_values("time_s").reset_index(drop=True)
    recent = df.tail(N_LAGS + 1)
    predicted_level = predict_congestion_level(model, feature_columns, recent)
    print(f"\nPredicted congestion level for the next interval: {predicted_level.upper()}")
    print("(This feeds directly into policy_generator.generate_candidate_policies(congestion_level=...))")
