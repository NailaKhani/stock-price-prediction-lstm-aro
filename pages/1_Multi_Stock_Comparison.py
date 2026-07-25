import sys
import os
# Add the project root to sys.path so that utils/ and models/ can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from utils.preprocessing import preprocess_data, forecast_future
from models.lstm_model import build_lstm_model
from utils.plot import plot_normalized_comparison, plot_correlation_heatmap

st.set_page_config(page_title="Multi-Stock Comparison", layout="wide")

st.title("Multi-Stock Comparison & Forecasting")
st.markdown("Compare historical performance and run lightweight fast-forecasts on multiple stocks simultaneously.")

# Configuration
with st.sidebar:
    st.header("Comparison Settings")
    tickers_input = st.text_input("Enter Tickers (comma separated)", "AAPL, MSFT, GOOGL")
    start_date = st.date_input("Start Date", pd.to_datetime('2022-01-01'))
    end_date = st.date_input("End Date", pd.to_datetime('today'))
    
    st.markdown("---")
    st.info("Note: Forecasting multiple stocks uses a lightweight Fast-LSTM without ARO optimization to keep the app responsive.")

# Parse tickers
tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

if not tickers:
    st.warning("Please enter at least one ticker symbol.")
else:
    # Fetch Data
    df_dict = {}
    with st.spinner("Fetching historical data..."):
        for ticker in tickers:
            try:
                df = yf.download(ticker, start=start_date, end=end_date, progress=False)
                if not df.empty:
                    df = df.reset_index()
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(0)
                    if 'Date' not in df.columns and 'Datetime' in df.columns:
                        df = df.rename(columns={'Datetime': 'Date'})
                    df_dict[ticker] = df
            except Exception as e:
                st.error(f"Failed to fetch data for {ticker}: {e}")

    if df_dict:
        # Display Growth Chart
        st.subheader("Historical Percentage Growth")
        st.markdown("This chart normalizes all stocks to 0% at the start date so you can directly compare their performance.")
        fig_growth = plot_normalized_comparison(df_dict)
        st.plotly_chart(fig_growth, use_container_width=True)
        
        # Display Correlation Heatmap
        if len(df_dict) > 1:
            st.markdown("---")
            st.subheader("Stock Correlation Heatmap")
            st.markdown("Shows how closely the stocks move together. 1.0 means perfect correlation, 0 means no correlation.")
            fig_corr = plot_correlation_heatmap(df_dict)
            st.plotly_chart(fig_corr, use_container_width=True)

        # Fast Forecasting
        st.markdown("---")
        st.subheader("Fast 30-Day Forecasting Comparison")
        
        if st.button("Run Fast Forecast for All Stocks"):
            forecast_results = {}
            
            # Progress bar for multiple stocks
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            import plotly.graph_objects as go
            fig_forecast = go.Figure()
            colors = ['#00f2fe', '#f6d365', '#ff0844', '#a78bfa', '#22c55e', '#f97316']
            
            for i, (ticker, df) in enumerate(df_dict.items()):
                status_text.text(f"Training Fast LSTM for {ticker}...")
                progress_bar.progress(i / len(df_dict))
                
                # Preprocess
                df_close = df[['Close']].copy()
                df_close['Close'] = pd.to_numeric(df_close['Close'].astype(str).str.replace(r'[\$,]', '', regex=True), errors='coerce')
                df_close.dropna(inplace=True)
                
                if len(df_close) < 50:
                    continue # Not enough data
                    
                X, y, scaler = preprocess_data(df_close)
                
                # Fast LSTM (small units, fewer epochs)
                model = build_lstm_model(X.shape[1:], units=32, dropout=0.2)
                model.fit(X, y, epochs=5, batch_size=32, verbose=0)
                
                # Forecast
                last_sequence = np.append(X[-1][1:], [y[-1]], axis=0).reshape(1, X.shape[1], 1)
                future_preds = forecast_future(model, scaler, last_sequence, n_days=30)
                future_flat = future_preds.flatten()
                
                # Build Date Axis
                last_date = df['Date'].iloc[-1] if 'Date' in df.columns else None
                if last_date is not None:
                    future_dates = pd.date_range(start=pd.Timestamp(last_date) + pd.Timedelta(days=1), periods=30, freq='B')
                    x_axis = future_dates
                else:
                    x_axis = list(range(30))
                
                # Add to plot
                fig_forecast.add_trace(go.Scatter(
                    x=x_axis, y=future_flat,
                    mode='lines', name=ticker,
                    line=dict(color=colors[i % len(colors)], width=2.5)
                ))
                
                forecast_results[ticker] = future_flat
                
            progress_bar.progress(1.0)
            status_text.text("Forecasting Complete!")
            
            fig_forecast.update_layout(
                title="Comparative 30-Day Price Forecast",
                xaxis_title="Date",
                yaxis_title="Predicted Price ($)",
                template="plotly_dark",
                hovermode="x unified",
                height=500,
                margin=dict(l=20, r=20, t=60, b=20),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_forecast, use_container_width=True)
