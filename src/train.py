import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from xgboost import XGBRegressor
import joblib, os

TARGET  = "aqi"
DROP    = ["city", "timestamp", "aqi_category", "weather", TARGET]

def load_features(path="data/features/features.parquet") -> pd.DataFrame:
    return pd.read_parquet(path)

def prepare(df: pd.DataFrame):
    feature_cols = [c for c in df.columns if c not in DROP]
    df = df[feature_cols + [TARGET]].dropna()
    X = df[feature_cols]
    y = df[TARGET]
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_xgboost(X_train, y_train) -> XGBRegressor:
    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train,
              eval_set=[(X_train, y_train)],
              verbose=50)
    return model

def evaluate(model, X_test, y_test):
    preds = model.predict(X_test)
    r2    = r2_score(y_test, preds)
    rmse  = np.sqrt(mean_squared_error(y_test, preds))
    mape  = np.mean(np.abs((y_test - preds) / (y_test + 1e-8))) * 100
    print(f"\n{'='*40}")
    print(f"  R²   : {r2:.4f}")
    print(f"  RMSE : {rmse:.4f}")
    print(f"  MAPE : {mape:.2f}%")
    print(f"{'='*40}\n")
    return {"r2": r2, "rmse": rmse, "mape": mape}

if __name__ == "__main__":
    df = load_features()
    X_train, X_test, y_train, y_test = prepare(df)
    print(f"Training on {len(X_train)} samples, testing on {len(X_test)}")
    model   = train_xgboost(X_train, y_train)
    metrics = evaluate(model, X_test, y_test)
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/xgboost_aqi.pkl")
    print("Model saved → models/xgboost_aqi.pkl")