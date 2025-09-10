# train_model.py (Corrected Version)
import pandas as pd
import numpy as np
import yfinance as yf
import ta
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error, r2_score

# SETTINGS
STOCK_SYMBOL = 'BHARTIARTL.NS'
START_DATE = '2010-01-01'
END_DATE = pd.Timestamp.now().strftime('%Y-%m-%d')
SEQUENCE_LENGTH = 60
MODEL_PATH = "enhanced_model.keras"
EPOCHS = 100
FEATURES = ['Close', 'SMA_20', 'EMA_20', 'RSI', 'BB_high', 'BB_low', 'MACD', 'Stoch_k']
TARGET_COL = 'Close'

# DATA GATHERING AND FEATURE ENGINEERING
print(f"Downloading data for {STOCK_SYMBOL}...")
df = yf.download(STOCK_SYMBOL, start=START_DATE, end=END_DATE)
print("Data download complete.")

print("Performing enhanced feature engineering...")
df['SMA_20'] = ta.trend.sma_indicator(df['Close'].squeeze(), window=20)
df['EMA_20'] = ta.trend.ema_indicator(df['Close'].squeeze(), window=20)
df['RSI'] = ta.momentum.rsi(df['Close'].squeeze(), window=14)
df['BB_high'] = ta.volatility.bollinger_hband(df['Close'].squeeze(), window=20)
df['BB_low'] = ta.volatility.bollinger_lband(df['Close'].squeeze(), window=20)
# --- FIX: Added .squeeze() to inputs ---
df['MACD'] = ta.trend.macd_diff(df['Close'].squeeze())
df['Stoch_k'] = ta.momentum.stoch(df['High'].squeeze(), df['Low'].squeeze(), df['Close'].squeeze(), window=14, smooth_window=3)

df.dropna(inplace=True)
print("Feature engineering complete.")

# (The rest of the script remains the same)
print("Preprocessing data...")
data = df[FEATURES]
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)
X, y = [], []
for i in range(SEQUENCE_LENGTH, len(scaled_data)):
    X.append(scaled_data[i-SEQUENCE_LENGTH:i])
    y.append(scaled_data[i, FEATURES.index(TARGET_COL)])
X, y = np.array(X), np.array(y)
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]
print("Data preprocessing complete.")
print("Building the enhanced LSTM model...")
model = Sequential([
    LSTM(units=100, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    LSTM(units=100, return_sequences=True),
    Dropout(0.2),
    LSTM(units=100, return_sequences=False),
    Dropout(0.2),
    Dense(units=25),
    Dense(units=1)
])
model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()
print("Model building complete.")
print(f"Training the model for {EPOCHS} epochs...")
history = model.fit(
    X_train, y_train, batch_size=32, epochs=EPOCHS, validation_split=0.1, verbose=1)
print("Model training complete.")
print(f"Saving enhanced model to {MODEL_PATH}...")
model.save(MODEL_PATH)
print("Model saved successfully.")