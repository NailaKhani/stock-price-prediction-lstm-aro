import streamlit as st
import pandas as pd
import numpy as np
import requests
from streamlit_lottie import st_lottie
from utils.preprocessing import preprocess_data, forecast_future
from models.lstm_model import build_lstm_model
from optimization.aro_optimizer import artificial_rabbit_optimization
from utils.metrics import evaluate
from utils.plot import plot_results, plot_candlestick_with_indicators, plot_future_forecast

@st.cache_data
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

@st.cache_data
def load_and_cache_data(uploaded_file):
    # Reset file pointer if needed, but st.cache_data handles UploadedFile correctly
    return pd.read_csv(uploaded_file)

# Set page configuration for a wider, more dashboard-like layout
st.set_page_config(page_title="Stock Predictor Pro", page_icon="📈", layout="wide")

# Custom CSS for Premium Glassmorphism UI
# st.markdown("""
# <style>
# ...
# </style>
# """, unsafe_allow_html=True)

st.title("📈 Stock Price Prediction with LSTM & ARO")
st.markdown("Predict future stock prices using a deep learning LSTM model, optimized by Artificial Rabbit Optimization (ARO).")

# Sidebar for controls
with st.sidebar:
    lottie_ai = load_lottieurl("https://assets9.lottiefiles.com/packages/lf20_hxart9lz.json")
    if lottie_ai:
        st_lottie(lottie_ai, height=150, key="ai_animation")
    st.header("⚙️ Configuration")
    uploaded_file = st.file_uploader("Upload your stock CSV file", type=["csv"])
    st.markdown("---")
    st.markdown("### Model Settings")
    n_iterations = st.slider("ARO Optimization Iterations", min_value=1, max_value=10, value=5)
    st.info("Higher iterations may take longer but can find better hyperparameters.")

if uploaded_file:
    df = load_and_cache_data(uploaded_file)
    
    st.markdown("---")
    st.subheader("📊 Interactive Candlestick Chart & Moving Averages")
    fig_candle = plot_candlestick_with_indicators(df)
    st.plotly_chart(fig_candle, use_container_width=True)
    
    # Strip spaces from columns and rename Close/Last to Close
    df.columns = [col.strip().replace('Close/Last', 'Close') for col in df.columns]
    
    # Dataset Summary Metrics
    st.markdown("---")
    st.subheader("📊 Dataset Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Records", len(df))
    col2.metric("Total Features", len(df.columns))
    
    if 'Close' in df.columns and pd.api.types.is_numeric_dtype(df['Close']):
        col3.metric("Latest Close Price", f"${df['Close'].iloc[-1]:.2f}")
    else:
        col3.metric("Latest Close Price", "N/A")
    
    with st.expander("🔍 View Raw Dataset Details"):
        st.write("Original DataFrame columns:", list(df.columns))
        st.dataframe(df.head(), use_container_width=True)

    # Extract Close column only and convert to numeric after removing $ and commas
    df_close = df[['Close']].copy()
    df_close['Close'] = pd.to_numeric(df_close['Close'].astype(str).str.replace('[\$,]', '', regex=True))
    
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
    st.subheader("🧠 Model Training & Optimization")
    
    def objective(params):
        model = build_lstm_model(X_train.shape[1:], params['units'], params['dropout'])
        model.fit(X_train, y_train, epochs=params['epochs'], batch_size=params['batch'], verbose=0)
        pred = model.predict(X_test, verbose=0)
        _, _, rmse = evaluate(y_test, pred)
        return rmse
    
    with st.spinner("⏳ Optimizing Hyperparameters using Artificial Rabbit Optimization (ARO)... This may take a few minutes."):
        best_params = artificial_rabbit_optimization(space, objective_fn=objective, n_iter=n_iterations)
    st.success(f"✅ Optimization Complete! Best Hyperparameters: {best_params}")
    
    with st.spinner("⏳ Training Final LSTM Model with best hyperparameters..."):
        # Train final model with best hyperparameters
        model = build_lstm_model(X_train.shape[1:], best_params['units'], best_params['dropout'])
        model.fit(X_train, y_train, epochs=best_params['epochs'], batch_size=best_params['batch'], verbose=0)
        pred = model.predict(X_test, verbose=0)
    
    # Evaluate model
    mae, mse, rmse = evaluate(y_test, pred)
    
    st.markdown("---")
    st.subheader("🎯 Evaluation Metrics")
    m_col1, m_col2, m_col3 = st.columns(3)
    m_col1.metric("Mean Absolute Error (MAE)", f"{mae:.4f}")
    m_col2.metric("Mean Squared Error (MSE)", f"{mse:.4f}")
    m_col3.metric("Root Mean Squared Error (RMSE)", f"{rmse:.4f}")
    
    # Future Forecasting
    st.markdown("---")
    st.subheader("🚀 Future Forecasting (Next 30 Days)")
    with st.spinner("⏳ Forecasting future stock prices..."):
        # Formulate last sequence from the entire dataset
        last_sequence = np.append(X[-1][1:], [y[-1]], axis=0)
        last_sequence = last_sequence.reshape(1, X_train.shape[1], 1)
        
        future_preds = forecast_future(model, scaler, last_sequence, n_days=30)
    
    st.subheader("📈 Prediction & Future Forecast Plot")
    fig = plot_future_forecast(y_test, pred, future_preds)
    st.plotly_chart(fig, use_container_width=True)
    
    # Download Button for Future Predictions
    future_df = pd.DataFrame(future_preds, columns=["Predicted Close Price"])
    future_df.index = [f"Day {i+1}" for i in range(30)]
    csv_data = future_df.to_csv().encode('utf-8')
    st.download_button(
        label="📥 Download 30-Day Future Predictions (CSV)",
        data=csv_data,
        file_name="future_predictions.csv",
        mime="text/csv"
    )

else:
    # Landing page state
    st.info("👈 Please upload a CSV file from the sidebar to get started.")
