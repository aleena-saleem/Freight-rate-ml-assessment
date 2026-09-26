import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv("train-test.csv")
df["date"] = pd.to_datetime(df["date"])

df.loc[df["weight"] <= 0, "weight"] = np.nan
df["weight"] = df["weight"].fillna(df["weight"].median())
df["market_index"] = df["market_index"].fillna(df["market_index"].median())

df["month"] = df["date"].dt.month
df["day"] = df["date"].dt.day
df["day_of_week"] = df["date"].dt.dayofweek
df["day_of_year"] = df["date"].dt.dayofyear
df["route"] = df["pickup"] + " -> " + df["delivery"]
df["distance_log"] = np.log1p(df["distance"])

features = [
    c for c in df.columns
    if c not in ["load_id", "posted_rate", "date"]
]

categorical = ["pickup", "delivery", "equipment", "route"]
categorical_indices = [features.index(c) for c in categorical]

splits = [
    ("Jan-Aug", df["date"] < "2025-09-01",
     (df["date"] >= "2025-09-01") & (df["date"] < "2025-11-01")),
    ("Jan-Sep", df["date"] < "2025-10-01",
     (df["date"] >= "2025-10-01") & (df["date"] < "2025-11-01"))
]

for name, train_mask, valid_mask in splits:
    train = df[train_mask]
    valid = df[valid_mask]

    model = CatBoostRegressor(
        iterations=1500,
        learning_rate=0.05,
        depth=8,
        loss_function="RMSE",
        random_seed=42,
        verbose=False
    )

    model.fit(
        train[features],
        train["posted_rate"],
        cat_features=categorical_indices,
        eval_set=(valid[features], valid["posted_rate"]),
        early_stopping_rounds=100
    )

    pred = model.predict(valid[features])

    mae = mean_absolute_error(valid["posted_rate"], pred)
    rmse = np.sqrt(mean_squared_error(valid["posted_rate"], pred))
    r2 = r2_score(valid["posted_rate"], pred)

    print(
        f"{name} | "
        f"MAE: {mae:.2f} | "
        f"RMSE: {rmse:.2f} | "
        f"R2: {r2:.4f}"
    )