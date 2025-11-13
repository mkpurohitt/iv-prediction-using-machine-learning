
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error

# 1. Load the intraday ATM IV dataset
df = pd.read_csv("intraday_atm_iv.csv", parse_dates=["timestamp"])

# 2. Resample to hourly IV (mean IV of each hour)
df = df.set_index("timestamp")
hourly_iv = df.groupby("symbol").resample("1H").mean().reset_index()

# 3. Create lag features
hourly_iv["iv_lag1"] = hourly_iv.groupby("symbol")["atm_iv"].shift(1)
hourly_iv["iv_lag3"] = hourly_iv.groupby("symbol")["atm_iv"].shift(1).rolling(3).mean()
hourly_iv["iv_lag6"] = hourly_iv.groupby("symbol")["atm_iv"].shift(1).rolling(6).mean()
hourly_iv["iv_change"] = hourly_iv.groupby("symbol")["atm_iv"].pct_change()

# 4. Drop NaNs
hourly_iv = hourly_iv.dropna().reset_index(drop=True)

predictions = []

# 5. Train/test and predict per symbol
for sym in hourly_iv["symbol"].unique():
    sym_df = hourly_iv[hourly_iv["symbol"] == sym].copy()

    X = sym_df[["iv_lag1", "iv_lag3", "iv_lag6", "iv_change"]]
    y = sym_df["atm_iv"]

    # Need enough rows
    if len(sym_df) < 50:
        print(f"Skipping {sym}, not enough data.")
        continue

    # Last 24 hours = test
    X_train, X_test = X.iloc[:-24], X.iloc[-24:]
    y_train, y_test = y.iloc[:-24], y.iloc[-24:]

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

    # Predict next hour
    next_input = X.iloc[[-1]]
    next_iv = model.predict(next_input)[0]

    predictions.append([
        sym,
        sym_df["timestamp"].iloc[-1] + pd.Timedelta(hours=1),
        next_iv
    ])

# 6. Save predictions
pred_df = pd.DataFrame(predictions, columns=["symbol", "pred_time", "predicted_atm_iv"])
pred_df.to_csv("next_hour_iv_predictions.csv", index=False)

print("✅ Predictions saved to next_hour_iv_predictions.csv")
print(pred_df)
