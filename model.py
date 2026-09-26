import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

train = pd.read_csv("train-test.csv")
validation = pd.read_csv("validation.csv")

train["date"] = pd.to_datetime(train["date"])
validation["date"] = pd.to_datetime(validation["date"])

for df in [train, validation]:
    df.loc[df["weight"] <= 0, "weight"] = np.nan

median_weight = train["weight"].median()
median_market = train["market_index"].median()

for df in [train, validation]:
    df["weight"] = df["weight"].fillna(median_weight)
    df["market_index"] = df["market_index"].fillna(median_market)

    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["day_of_week"] = df["date"].dt.dayofweek
    df["day_of_year"] = df["date"].dt.dayofyear
    df["route"] = df["pickup"] + " -> " + df["delivery"]
    df["distance_log"] = np.log1p(df["distance"])

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

predictions = model.predict(validation[features])

output = pd.DataFrame({
    "load_id": validation["load_id"],
    "predicted_rate": predictions
})

output["predicted_rate"] = output["predicted_rate"].clip(lower=0.01)

output.to_csv(
    "validation_predictions.csv",
    index=False
)