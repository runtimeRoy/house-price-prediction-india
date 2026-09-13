"""
House Price Prediction - Indian Housing Dataset
------------------------------------------------
Trains Linear Regression and Ridge Regression models on the
House Price India Kaggle dataset.

Run:
    python house_price_prediction.py
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "House Price India.csv"
OUTPUT_PATH = BASE_DIR / "house_price_prediction.png"

TARGET = "Price"

df = pd.read_csv(DATA_PATH)

# Drop identifiers / date-like fields that are not useful for the basic model.
DROP_COLUMNS = [c for c in ["id", "Date"] if c in df.columns]
df = df.drop(columns=DROP_COLUMNS)

# Keep numeric features and clean missing/infinite values.
X = df.drop(columns=[TARGET])
X = X.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan)
X = X.fillna(X.median(numeric_only=True))
y = pd.to_numeric(df[TARGET], errors="coerce")

valid = y.notna()
X, y = X.loc[valid], y.loc[valid]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
}

results = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    predictions = model.predict(X_test_scaled)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)
    cv_r2 = cross_val_score(
        model, scaler.transform(X), y, cv=5, scoring="r2"
    ).mean()
    results[name] = {"model": model, "rmse": rmse, "r2": r2, "cv_r2": cv_r2}

print("\nIndian House Price Prediction Results")
print("=" * 45)
for name, result in results.items():
    print(f"{name}:")
    print(f"  RMSE   : {result['rmse']:,.2f}")
    print(f"  R²     : {result['r2']:.4f}")
    print(f"  5-Fold CV R²: {result['cv_r2']:.4f}")

# Visualization: actual vs predicted for Ridge Regression.
ridge_predictions = results["Ridge Regression"]["model"].predict(X_test_scaled)

plt.figure(figsize=(9, 6))
plt.scatter(y_test, ridge_predictions, alpha=0.45)
line_min = min(y_test.min(), ridge_predictions.min())
line_max = max(y_test.max(), ridge_predictions.max())
plt.plot([line_min, line_max], [line_min, line_max], linestyle="--")
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Indian House Price Prediction - Ridge Regression")
plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=160)
plt.show()