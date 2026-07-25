import streamlit as st
import pandas as pd
import numpy as np
import requests
import yfinance as yf
from datetime import datetime
from streamlit_lottie import st_lottie
from utils.preprocessing import preprocess_data, forecast_future
from models.lstm_model import build_lstm_model
from optimization.aro_optimizer import artificial_rabbit_optimization
from utils.metrics import evaluate
import utils.plot
from utils.plot import plot_results, plot_candlestick_with_indicators, plot_future_forecast

@st.cache_data
def load_lottieurl(url: str):
    try:
        r = requests.get(url, timeout=5)
    except Exception:
        return None
    if r.status_code != 200:
        return None
    return r.json()

@st.cache_data
def load_and_cache_data(uploaded_file):
    # Reset file pointer if needed, but st.cache_data handles UploadedFile correctly
    return pd.read_csv(uploaded_file)

# Set page configuration for a wider, more dashboard-like layout
st.set_page_config(page_title="Stock Predictor Pro", layout="wide")

# Custom CSS for Premium Glassmorphism UI
# st.markdown("""
# <style>
# ...
# </style>
# """, unsafe_allow_html=True)

st.title("Stock Price Prediction with LSTM & ARO")
st.markdown("Predict future stock prices using a deep learning LSTM model, optimized by Artificial Rabbit Optimization (ARO).")

# Sidebar for controls
with st.sidebar:
    lottie_ai = load_lottieurl("https://assets9.lottiefiles.com/packages/lf20_hxart9lz.json")
    if lottie_ai:
        st_lottie(lottie_ai, height=150, key="ai_animation")
    st.header("Configuration")
    data_source = st.radio("Select Data Source", ["Live Data (Yahoo Finance)", "CSV Upload"])
    
    uploaded_file = None
    ticker_symbol = None
    
    if data_source == "CSV Upload":
        uploaded_file = st.file_uploader("Upload your stock CSV file", type=["csv"])
    else:
        ticker_symbol = st.text_input("Enter Stock Ticker (e.g., AAPL, TSLA)", "AAPL")
        start_date = st.date_input("Start Date", pd.to_datetime('2020-01-01'))
        end_date = st.date_input("End Date", pd.to_datetime('today'))
        
    st.markdown("---")
    st.markdown("### Model Settings")
    n_iterations = st.slider("ARO Optimization Iterations", min_value=1, max_value=10, value=5)
    st.info("Higher iterations may take longer but can find better hyperparameters.")

df = None
if data_source == "CSV Upload" and uploaded_file:
    df = load_and_cache_data(uploaded_file)
    # Strip spaces from columns and rename Close/Last to Close
    df.columns = [col.strip().replace('Close/Last', 'Close') for col in df.columns]
elif data_source == "Live Data (Yahoo Finance)" and ticker_symbol:
    with st.spinner(f"Fetching live data for {ticker_symbol}..."):
        df = yf.download(ticker_symbol, start=start_date, end=end_date, progress=False)
        if not df.empty:
            df = df.reset_index()
            
            # yfinance returns MultiIndex columns sometimes in newer versions, flatten them if needed
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
                
            # Rename for consistency if necessary
            if 'Date' not in df.columns and 'Datetime' in df.columns:
                df = df.rename(columns={'Datetime': 'Date'})

if df is not None and not df.empty:
    
    st.markdown("---")
    st.subheader("Advanced Technical Indicators Dashboard")
    fig_candle = plot_candlestick_with_indicators(df)
    st.plotly_chart(fig_candle, use_container_width=True)
    
    # Dataset Summary Metrics
    st.markdown("---")
    st.subheader("Dataset Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Records", len(df))
    col2.metric("Total Features", len(df.columns))
    
    if 'Close' in df.columns:
        close_numeric = pd.to_numeric(df['Close'], errors='coerce')
        last_val = close_numeric.dropna().iloc[-1] if not close_numeric.dropna().empty else None
        col3.metric("Latest Close Price", f"${last_val:.2f}" if last_val is not None else "N/A")
    else:
        col3.metric("Latest Close Price", "N/A")
    
    with st.expander("View Raw Dataset Details"):
        st.write("Original DataFrame columns:", list(df.columns))
        st.dataframe(df.head(), use_container_width=True)

    # Extract Close column only and convert to numeric after removing $ and commas
    df_close = df[['Close']].copy()
    df_close['Close'] = pd.to_numeric(df_close['Close'].astype(str).str.replace('[\$,]', '', regex=True), errors='coerce')
    df_close.dropna(inplace=True)
    
    # Preprocess data with only Close prices
    X, y, scaler = preprocess_data(df_close)
    
    # Train/test split
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Hyperparameter space
    space = {
        "units": [32, 50, 64],
        "dropout": [0.2, 0.3],
        "batch": [16, 32],
        "epochs": [10, 20]
    }
    
    st.markdown("---")
    st.subheader("Model Training & Optimization")
    
    def objective(params):
        model = build_lstm_model(X_train.shape[1:], params['units'], params['dropout'])
        model.fit(X_train, y_train, epochs=params['epochs'], batch_size=params['batch'], verbose=0)
        pred = model.predict(X_test, verbose=0)
        _, _, rmse = evaluate(y_test, pred)
        return rmse
    
    with st.spinner("Optimizing Hyperparameters using Artificial Rabbit Optimization (ARO)... This may take a few minutes."):
        best_params = artificial_rabbit_optimization(space, objective_fn=objective, n_iter=n_iterations)
    st.success(f"Optimization Complete! Best Hyperparameters: {best_params}")
    
    with st.spinner("Training Final LSTM Model with best hyperparameters..."):
        # Train final model with best hyperparameters
        model = build_lstm_model(X_train.shape[1:], best_params['units'], best_params['dropout'])
        model.fit(X_train, y_train, epochs=best_params['epochs'], batch_size=best_params['batch'], verbose=0)
        pred = model.predict(X_test, verbose=0)
    
    # Inverse transform predictions to real dollar values
    y_test_real = scaler.inverse_transform(y_test)
    pred_real = scaler.inverse_transform(pred)

    # Evaluate model on scaled data (standard practice)
    mae, mse, rmse = evaluate(y_test, pred)
    # Also compute real-scale MAE for display
    mae_real, mse_real, rmse_real = evaluate(y_test_real, pred_real)
    
    st.markdown("---")
    st.subheader("🎯 Evaluation Metrics")
    m_col1, m_col2, m_col3 = st.columns(3)
    m_col1.metric("Mean Absolute Error (MAE)", f"${mae_real:.2f}")
    m_col2.metric("Mean Squared Error (MSE)", f"${mse_real:.2f}")
    m_col3.metric("Root Mean Squared Error (RMSE)", f"${rmse_real:.2f}")
    
    # ── Phase 3: Enhanced Future Forecasting ──────────────────────────────
    st.markdown("---")
    st.subheader("Enhanced Future Forecasting (Next 30 Days)")
    with st.spinner("Forecasting future stock prices..."):
        last_sequence = np.append(X[-1][1:], [y[-1]], axis=0)
        last_sequence = last_sequence.reshape(1, X_train.shape[1], 1)
        future_preds = forecast_future(model, scaler, last_sequence, n_days=30)

    # Determine last known real date from dataframe
    last_date = None
    if 'Date' in df.columns:
        last_date = pd.to_datetime(df['Date']).dropna().iloc[-1]

    # Plot with real dates + confidence band
    fig = plot_future_forecast(y_test_real, pred_real, future_preds, last_date=last_date)
    st.plotly_chart(fig, use_container_width=True)

    # ── Forecast Summary Table ────────────────────────────────────────────
    st.markdown("#### 30-Day Day-by-Day Forecast")
    if last_date is not None:
        future_dates_range = pd.date_range(
            start=pd.Timestamp(last_date) + pd.Timedelta(days=1), periods=30, freq='B'
        )
        date_labels = [d.strftime('%Y-%m-%d') for d in future_dates_range]
    else:
        date_labels = [f"Day {i+1}" for i in range(30)]

    future_flat = future_preds.flatten()
    last_real_price = float(y_test_real.flatten()[-1])
    forecast_rows = []
    for i, (date, price) in enumerate(zip(date_labels, future_flat)):
        prev = last_real_price if i == 0 else float(future_flat[i - 1])
        change = price - prev
        change_pct = (change / prev) * 100 if prev != 0 else 0
        signal = "Buy" if change > 0 else "Sell"
        forecast_rows.append({
            "Date": date,
            "Forecast Price ($)": f"${price:.2f}",
            "Change ($)": f"{'+' if change >= 0 else ''}{change:.2f}",
            "Change (%)": f"{'+' if change_pct >= 0 else ''}{change_pct:.2f}%",
            "Signal": signal
        })

    st.dataframe(pd.DataFrame(forecast_rows), use_container_width=True, height=280)

    # ── Summary Metrics ───────────────────────────────────────────────────
    max_price = float(future_flat.max())
    min_price = float(future_flat.min())
    avg_price = float(future_flat.mean())
    trend_label = "Bullish" if future_flat[-1] > future_flat[0] else "Bearish"
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Peak Forecast", f"${max_price:.2f}")
    s2.metric("Trough Forecast", f"${min_price:.2f}")
    s3.metric("Avg Forecast", f"${avg_price:.2f}")
    s4.metric("30-Day Trend", trend_label)

    # ── Download ──────────────────────────────────────────────────────────
    st.markdown("---")
    download_df = pd.DataFrame({"Date": date_labels, "Predicted Close Price ($)": future_flat})
    csv_data = download_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download 30-Day Forecast as CSV",
        data=csv_data,
        file_name=f"{ticker_symbol or 'stock'}_30day_forecast.csv",
        mime="text/csv"
    )

else:
    # Landing page state
    if data_source == "CSV Upload":
        st.info("Please upload a CSV file from the sidebar to get started.")
    else:
        st.info("Enter a valid Stock Ticker in the sidebar to get started.")
