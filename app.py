# app.py (Final Polished Version)
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import ta
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error, r2_score
import plotly.graph_objects as go

# --- SETTINGS ---
MODEL_PATH = "enhanced_model.keras"
SEQ_LEN = 60
FEATURES = ['Close', 'SMA_20', 'EMA_20', 'RSI', 'BB_high', 'BB_low', 'MACD', 'Stoch_k']
TARGET_COL = 'Close'

# --- PAGE CONFIG AND STYLING ---
st.set_page_config(page_title="📈 Advanced Stock Forecast", layout="wide")
st.markdown("<h1 style='text-align:center;color:#4CAF50;'>📈 Advanced Stock Price Forecast</h1>", unsafe_allow_html=True)
st.markdown("""<style>
/* Metric card styling */
[data-testid="stMetric"] { background-color: #262730; border: 1px solid #262730; padding: 20px; border-radius: 10px; color: white; box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2); }
/* Align button vertically with the selectbox */
div.stButton > button { margin-top: 28px; }
</style>""", unsafe_allow_html=True)

# --- CACHED DATA FUNCTIONS ---
@st.cache_data
def load_tickers():
    try:
        df = pd.read_csv('nse_tickers.csv')
        df['DISPLAY'] = df['COMPANY_NAME'] + " (" + df['SYMBOL'] + ")"
        return df
    except FileNotFoundError:
        return None

@st.cache_data
def get_stock_info(ticker_symbol):
    """Fetches key information for a given stock symbol."""
    try:
        ticker = yf.Ticker(ticker_symbol)
        info = ticker.info
        return info
    except Exception:
        return None

@st.cache_data
def load_and_prepare_data(stock_symbol):
    end_date = pd.Timestamp.now().strftime('%Y-%m-%d')
    data_df = yf.download(stock_symbol, start='2010-01-01', end=end_date)
    if data_df.empty: return None, None, None, None, None, None
    data_df['SMA_20'] = ta.trend.sma_indicator(data_df['Close'].squeeze(), window=20)
    data_df['EMA_20'] = ta.trend.ema_indicator(data_df['Close'].squeeze(), window=20)
    data_df['RSI'] = ta.momentum.rsi(data_df['Close'].squeeze(), window=14)
    data_df['BB_high'] = ta.volatility.bollinger_hband(data_df['Close'].squeeze(), window=20)
    data_df['BB_low'] = ta.volatility.bollinger_lband(data_df['Close'].squeeze(), window=20)
    data_df['MACD'] = ta.trend.macd_diff(data_df['Close'].squeeze())
    data_df['Stoch_k'] = ta.momentum.stoch(data_df['High'].squeeze(), data_df['Low'].squeeze(), data_df['Close'].squeeze(), window=14, smooth_window=3)
    data_df.dropna(inplace=True)
    data = data_df[FEATURES]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    X, y = [], []
    for i in range(SEQ_LEN, len(scaled_data)):
        X.append(scaled_data[i-SEQ_LEN:i])
        y.append(scaled_data[i, FEATURES.index(TARGET_COL)])
    X, y = np.array(X), np.array(y)
    split = int(0.8 * len(X))
    X_test, y_test = X[split:], y[split:]
    test_dates = data_df.index[SEQ_LEN + split:]
    return X_test, y_test, scaler, test_dates, scaled_data, data_df

# --- INPUT CONTROLS ---
tickers_df = load_tickers()
st.subheader("⚙️ Model Parameters")
col1, col2 = st.columns([3, 1])

with col1:
    if tickers_df is not None:
        try:
            default_index = int(tickers_df[tickers_df['SYMBOL'] == 'BHARTIARTL.NS'].index[0])
        except (IndexError, TypeError):
            default_index = 0
        selected_display = st.selectbox('Search for a stock', options=tickers_df['DISPLAY'], index=default_index)
        stock = tickers_df[tickers_df['DISPLAY'] == selected_display]['SYMBOL'].iloc[0]
    else:
        stock = st.text_input('Enter Stock Symbol', 'BHARTIARTL.NS')

with col2:
    run_button = st.button('Run Forecast', use_container_width=True, type="primary")

future_days = st.slider('Days to Forecast', 1, 15, 5)
st.divider()

# --- MAIN APP LOGIC ---
if run_button:
    with st.spinner('Running forecast... Please wait.'):
        try:
            X_test, y_test, scaler, test_dates, scaled_data, data_df = load_and_prepare_data(stock)
            if X_test is None:
                st.error(f"🚫 No data fetched for {stock}.")
            else:
                model = load_model(MODEL_PATH)

                # --- NEW: STOCK INFORMATION SECTION (NO LOGO) ---
                st.subheader(f"🏢 Company Profile & Key Metrics")
                stock_info = get_stock_info(stock)
                if stock_info:
                    st.markdown(f"**{stock_info.get('longName', 'N/A')}** ({stock_info.get('symbol', 'N/A')})")
                    st.markdown(f"**Sector**: {stock_info.get('sector', 'N/A')} | **Industry**: {stock_info.get('industry', 'N/A')}")

                    metric_col1, metric_col2, metric_col3 = st.columns(3)
                    market_cap = stock_info.get('marketCap', 0)
                    high_52 = stock_info.get('fiftyTwoWeekHigh', 0)
                    low_52 = stock_info.get('fiftyTwoWeekLow', 0)

                    metric_col1.metric("Market Cap", f"₹{market_cap:,.0f}" if market_cap else "N/A")
                    metric_col2.metric("52 Week High", f"₹{high_52:,.2f}" if high_52 else "N/A")
                    metric_col3.metric("52 Week Low", f"₹{low_52:,.2f}" if low_52 else "N/A")

                    with st.expander("Business Summary"):
                        st.write(stock_info.get('longBusinessSummary', 'No summary available.'))
                else:
                    st.warning("Could not retrieve detailed stock information.")
                st.divider()
                # ------------------------------------

                # Backtesting
                st.subheader(f"📊 Backtesting Performance")
                predictions_scaled = model.predict(X_test)
                dummy_array = np.zeros((len(predictions_scaled), len(FEATURES)))
                dummy_array[:, FEATURES.index(TARGET_COL)] = predictions_scaled.flatten()
                predicted_prices = scaler.inverse_transform(dummy_array)[:, FEATURES.index(TARGET_COL)]
                dummy_array_actual = np.zeros((len(y_test), len(FEATURES)))
                dummy_array_actual[:, FEATURES.index(TARGET_COL)] = y_test.flatten()
                actual_prices = scaler.inverse_transform(dummy_array_actual)[:, FEATURES.index(TARGET_COL)]
                mape = mean_absolute_percentage_error(actual_prices, predicted_prices) * 100
                accuracy = 100 - mape
                r_squared = r2_score(actual_prices, predicted_prices)

                perf_col1, perf_col2, perf_col3 = st.columns(3)
                perf_col1.metric("📊 Model Accuracy", f"{accuracy:.2f}%")
                perf_col2.metric("📉 MAPE", f"{mape:.2f}%")
                perf_col3.metric("📈 R² Score", f"{r_squared:.4f}")

                st.divider()

                # Iterative Forecast
                last_sequence = scaled_data[-SEQ_LEN:]
                current_batch = last_sequence.reshape(1, SEQ_LEN, len(FEATURES))
                future_predictions_scaled = []
                for i in range(future_days):
                    next_prediction_scaled = model.predict(current_batch)[0]
                    future_predictions_scaled.append(next_prediction_scaled)
                    new_step = current_batch[0, -1, :].copy()
                    new_step[FEATURES.index(TARGET_COL)] = next_prediction_scaled
                    current_batch = np.append(current_batch[:, 1:, :], [[new_step]], axis=1)
                dummy_array_future = np.zeros((len(future_predictions_scaled), len(FEATURES)))
                for i, pred in enumerate(future_predictions_scaled):
                    dummy_array_future[i, FEATURES.index(TARGET_COL)] = pred
                future_predicted_prices = scaler.inverse_transform(dummy_array_future)[:, FEATURES.index(TARGET_COL)]
                last_date = data_df.index[-1]
                future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=future_days)

                # Visualization
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=test_dates, y=actual_prices, mode='lines', name='Actual Price', line=dict(color='royalblue')))
                fig.add_trace(go.Scatter(x=test_dates, y=predicted_prices, mode='lines', name='Predicted Price (Backtest)', line=dict(color='firebrick', dash='dot')))
                fig.add_trace(go.Scatter(x=future_dates, y=future_predicted_prices, mode='lines+markers', name='Future Forecast', line=dict(color='gold', width=3, dash='dash')))
                fig.update_layout(title=f'📈 Stock Price Analysis & {future_days}-Day Forecast for {stock}', xaxis_title='Date', yaxis_title='Price', template='plotly_dark')
                st.plotly_chart(fig, use_container_width=True)
                st.balloons()

        except FileNotFoundError:
            st.error(f"Model file not found at '{MODEL_PATH}'. Please run train_model.py first.")
        except Exception as e:
            st.error(f"An error occurred: {e}")
else:
    st.info("Please select your parameters and click 'Run Forecast'.")

