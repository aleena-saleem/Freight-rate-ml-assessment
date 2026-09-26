import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

train = pd.read_csv("train-test.csv")
train["date"] = pd.to_datetime(train["date"])

train.loc[train["weight"] <= 0, "weight"] = np.nan
train["weight"] = train["weight"].fillna(train["weight"].median())
train["market_index"] = train["market_index"].fillna(train["market_index"].median())

train["month"] = train["date"].dt.month
train["day"] = train["date"].dt.day
train["day_of_week"] = train["date"].dt.dayofweek
train["day_of_year"] = train["date"].dt.dayofyear
train["route"] = train["pickup"] + " -> " + train["delivery"]
train["distance_log"] = np.log1p(train["distance"])

features = [
    c for c in train.columns
    if c not in ["load_id", "posted_rate", "date"]
]

categorical = ["pickup", "delivery", "equipment", "route"]
categorical_indices = [features.index(c) for c in categorical]

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
    cat_features=categorical_indices
)

route = train[
    (train["pickup"] == "Lexington") &
    (train["delivery"] == "Fort Wayne")
]

if len(route) == 0:
    raise ValueError("Lexington to Fort Wayne route not found in training data")

pickup_lat = route["pickup_lat"].median()
pickup_lon = route["pickup_lon"].median()
delivery_lat = route["delivery_lat"].median()
delivery_lon = route["delivery_lon"].median()

december = pd.DataFrame({
    "pickup": "Lexington",
    "delivery": "Fort Wayne",
    "pickup_lat": pickup_lat,
    "pickup_lon": pickup_lon,
    "delivery_lat": delivery_lat,
    "delivery_lon": delivery_lon,
    "distance": 360.0,
    "equipment": "Dry Van",
    "weight": 32000.0,
    "date": pd.date_range("2025-12-01", "2025-12-31")
})

december["market_index"] = train["market_index"].median()
december["quote_signal"] = train["quote_signal"].median()
december["month"] = december["date"].dt.month
december["day"] = december["date"].dt.day
december["day_of_week"] = december["date"].dt.dayofweek
december["day_of_year"] = december["date"].dt.dayofyear
december["route"] = december["pickup"] + " -> " + december["delivery"]
december["distance_log"] = np.log1p(december["distance"])

december["predicted_rate"] = model.predict(december[features])
december["predicted_rate"] = december["predicted_rate"].clip(lower=0.01)

output = december[
    [
        "pickup",
        "delivery",
        "distance",
        "equipment",
        "weight",
        "date",
        "predicted_rate"
    ]
]

output.to_csv("december_chart_inputs.csv", index=False)