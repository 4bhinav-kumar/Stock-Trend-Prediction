import yfinance as yf
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# --------------------------------
# 1. Select Stock
# --------------------------------

ticker = "AAPL"


# --------------------------------
# 2. Download Historical Data
# --------------------------------

print("Downloading stock data...")

data = yf.download(
    ticker,
    period="5y",
    auto_adjust=True
)


# --------------------------------
# 3. Fix column structure
# --------------------------------

if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)


# --------------------------------
# 4. Feature Engineering
# --------------------------------

data["MA5"] = data["Close"].rolling(5).mean()

data["MA20"] = data["Close"].rolling(20).mean()

data["Return"] = data["Close"].pct_change()


# --------------------------------
# 5. Create Target
# --------------------------------

data["Target"] = (
    data["Close"].shift(-1) > data["Close"]
).astype(int)


# --------------------------------
# 6. Remove missing values
# --------------------------------

data.dropna(inplace=True)


# --------------------------------
# 7. Select Features
# --------------------------------

features = [
    "Close",
    "Volume",
    "MA5",
    "MA20",
    "Return"
]

X = data[features]

y = data["Target"]


# --------------------------------
# 8. Train-Test Split
# --------------------------------

split = int(len(data) * 0.8)

X_train = X.iloc[:split]

X_test = X.iloc[split:]

y_train = y.iloc[:split]

y_test = y.iloc[split:]


# --------------------------------
# 9. Create ML Model
# --------------------------------

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)


# --------------------------------
# 10. Train Model
# --------------------------------

print("Training model...")

model.fit(
    X_train,
    y_train
)


# --------------------------------
# 11. Prediction
# --------------------------------

prediction = model.predict(X_test)


# --------------------------------
# 12. Accuracy
# --------------------------------

accuracy = accuracy_score(
    y_test,
    prediction
)

print(
    "Model Accuracy:",
    accuracy
)


# --------------------------------
# 13. Save Model
# --------------------------------

joblib.dump(
    model,
    "model.pkl"
)

print("Model saved successfully!")

print("File created: model.pkl")