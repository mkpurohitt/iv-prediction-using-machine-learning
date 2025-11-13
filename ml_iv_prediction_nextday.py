import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error

# Load dataset
df = pd.read_csv("atm_iv_dataset.csv", parse_dates=["date"])
df = df.sort_values(["symbol", "date"]).reset_index(drop=True)

# Create lag features
df["iv_lag1"] = df.groupby("symbol")["atm_iv"].shift(1)
df["iv_lag3"] = df.groupby("symbol")["atm_iv"].shift(1).rolling(3).mean()
df["iv_lag5"] = df.groupby("symbol")["atm_iv"].shift(1).rolling(5).mean()
df["iv_change"] = df.groupby("symbol")["atm_iv"].pct_change()

# Drop missing values safely
df = df.dropna().reset_index(drop=True)

predictions = []

for sym in df["symbol"].unique():
    sym_df = df[df["symbol"] == sym].copy()

    X = sym_df[["iv_lag1", "iv_lag3", "iv_lag5", "iv_change"]]
    y = sym_df["atm_iv"]

    # If not enough rows, skip
    if len(sym_df) < 50:
        print(f"Skipping {sym}, not enough data.")
        continue

    # Train/test split (last 30 rows as test)
    X_train, X_test = X.iloc[:-30], X.iloc[-30:]
    y_train, y_test = y.iloc[:-30], y.iloc[-30:]

    if X_train.empty or X_test.empty:
        print(f"Skipping {sym}, empty train/test split.")
        continue

    # Train LightGBM
    model = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"{sym} Test MAE: {mae:.4f}")

    # Predict next day
    next_input = X.iloc[[-1]]
    next_iv = model.predict(next_input)[0]

    predictions.append([sym, sym_df["date"].iloc[-1] + pd.Timedelta(days=1), next_iv])

# Save predictions
pred_df = pd.DataFrame(predictions, columns=["symbol", "pred_date", "predicted_atm_iv"])
pred_df.to_csv("next_day_iv_predictions.csv", index=False)

print("✅ Predictions saved to next_day_iv_predictions.csv")
print(pred_df)




