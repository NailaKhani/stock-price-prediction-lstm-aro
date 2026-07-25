import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import timedelta


def plot_results(y_true, y_pred):
    fig = go.Figure()

    y_true_flat = y_true.flatten() if hasattr(y_true, 'flatten') else y_true
    y_pred_flat = y_pred.flatten() if hasattr(y_pred, 'flatten') else y_pred

    fig.add_trace(go.Scatter(
        y=y_true_flat,
        mode='lines',
        name='Actual Price',
        line=dict(color='#00f2fe', width=3),
        fill='tozeroy',
        fillcolor='rgba(0, 242, 254, 0.1)'
    ))

    fig.add_trace(go.Scatter(
        y=y_pred_flat,
        mode='lines',
        name='Predicted Price',
        line=dict(color='#ff0844', width=3, dash='dot')
    ))

    fig.update_layout(
        title=dict(text="Prediction Comparison", font=dict(size=24, color='#ffffff')),
        xaxis_title="Time (Days)",
        yaxis_title="Stock Price ($)",
        template="plotly_dark",
        hovermode="x unified",
        margin=dict(l=20, r=20, t=60, b=20),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)')
    )
    return fig


def plot_candlestick_with_indicators(df):
    # Normalize column names
    df_plot = df.copy()
    df_plot.columns = [col.strip() for col in df_plot.columns]
    df_plot.rename(columns={'Close/Last': 'Close'}, inplace=True)
    cols_lower = [col.lower() for col in df_plot.columns]

    if not all(x in cols_lower for x in ['open', 'high', 'low', 'close']):
        # Fallback if no OHLC
        fig = go.Figure()
        return fig

    for c in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if c in df_plot.columns:
            df_plot[c] = pd.to_numeric(
                df_plot[c].astype(str).str.replace(r'[\$,]', '', regex=True),
                errors='coerce'
            )
    df_plot.dropna(subset=['Open', 'High', 'Low', 'Close'], inplace=True)

    if 'Date' in df_plot.columns:
        df_plot['Date'] = pd.to_datetime(df_plot['Date'], errors='coerce')
        x_axis = df_plot['Date']
    elif 'Datetime' in df_plot.columns:
        df_plot['Datetime'] = pd.to_datetime(df_plot['Datetime'], errors='coerce')
        x_axis = df_plot['Datetime']
    else:
        x_axis = df_plot.index

    # Moving Averages
    df_plot['SMA20'] = df_plot['Close'].rolling(window=20).mean()
    df_plot['EMA20'] = df_plot['Close'].ewm(span=20, adjust=False).mean()

    # RSI (14 days)
    delta = df_plot['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
    rs = gain / loss
    df_plot['RSI'] = 100 - (100 / (1 + rs))

    # MACD
    exp1 = df_plot['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df_plot['Close'].ewm(span=26, adjust=False).mean()
    df_plot['MACD'] = exp1 - exp2
    df_plot['Signal_Line'] = df_plot['MACD'].ewm(span=9, adjust=False).mean()

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        vertical_spacing=0.05,
                        row_heights=[0.6, 0.2, 0.2])

    # Candlestick
    fig.add_trace(go.Candlestick(x=x_axis, open=df_plot['Open'], high=df_plot['High'],
                                  low=df_plot['Low'], close=df_plot['Close'], name='Price'), row=1, col=1)
    fig.add_trace(go.Scatter(x=x_axis, y=df_plot['SMA20'],
                              line=dict(color='orange', width=1.5), name='SMA 20'), row=1, col=1)
    fig.add_trace(go.Scatter(x=x_axis, y=df_plot['EMA20'],
                              line=dict(color='cyan', width=1.5), name='EMA 20'), row=1, col=1)

    # RSI
    fig.add_trace(go.Scatter(x=x_axis, y=df_plot['RSI'],
                              line=dict(color='#a78bfa', width=1.5), name='RSI'), row=2, col=1)
    fig.add_hline(y=70, line_dash="dot", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color="green", row=2, col=1)

    # MACD
    fig.add_trace(go.Scatter(x=x_axis, y=df_plot['MACD'],
                              line=dict(color='#60a5fa', width=1.5), name='MACD'), row=3, col=1)
    fig.add_trace(go.Scatter(x=x_axis, y=df_plot['Signal_Line'],
                              line=dict(color='orange', width=1.5), name='Signal'), row=3, col=1)
    macd_hist = df_plot['MACD'] - df_plot['Signal_Line']
    fig.add_trace(go.Bar(x=x_axis, y=macd_hist, name='MACD Hist',
                          marker_color=['#22c55e' if v >= 0 else '#ef4444' for v in macd_hist]), row=3, col=1)

    fig.update_layout(
        title="Advanced Technical Indicators Dashboard",
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        height=800,
        margin=dict(l=20, r=20, t=60, b=20),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )
    fig.update_yaxes(title_text="Price ($)", row=1, col=1, showgrid=True, gridcolor='rgba(255,255,255,0.1)')
    fig.update_yaxes(title_text="RSI", row=2, col=1, showgrid=True, gridcolor='rgba(255,255,255,0.1)', range=[0, 100])
    fig.update_yaxes(title_text="MACD", row=3, col=1, showgrid=True, gridcolor='rgba(255,255,255,0.1)')
    fig.update_xaxes(showgrid=True, gridcolor='rgba(255,255,255,0.08)')

    return fig


def plot_future_forecast(y_true, y_pred, future_preds, last_date=None, n_future=30):
    y_true_flat = y_true.flatten() if hasattr(y_true, 'flatten') else y_true
    y_pred_flat = y_pred.flatten() if hasattr(y_pred, 'flatten') else y_pred
    future_flat = future_preds.flatten() if hasattr(future_preds, 'flatten') else future_preds

    # Build real date x-axis if last_date is given, else use integer index
    if last_date is not None:
        hist_dates = pd.date_range(end=last_date, periods=len(y_true_flat), freq='B')
        future_dates = pd.date_range(
            start=pd.Timestamp(last_date) + timedelta(days=1),
            periods=len(future_flat), freq='B'
        )
        x_hist = list(hist_dates)
        x_future = list(future_dates)
        x_connect = [hist_dates[-1], future_dates[0]]
        y_connect = [float(y_true_flat[-1]), float(future_flat[0])]
        forecast_start_x = future_dates[0]
    else:
        x_hist = list(range(len(y_true_flat)))
        x_future = list(range(len(y_true_flat), len(y_true_flat) + len(future_flat)))
        x_connect = [len(y_true_flat) - 1, len(y_true_flat)]
        y_connect = [float(y_true_flat[-1]), float(future_flat[0])]
        forecast_start_x = len(y_true_flat)

    # Confidence band (~5% spread as a simple uncertainty estimate)
    upper_band = future_flat * 1.05
    lower_band = future_flat * 0.95

    fig = go.Figure()

    # Actual Price area
    fig.add_trace(go.Scatter(
        x=x_hist, y=y_true_flat,
        mode='lines', name='Actual Price',
        line=dict(color='#00f2fe', width=2.5),
        fill='tozeroy', fillcolor='rgba(0, 242, 254, 0.07)',
        hovertemplate='<b>Actual:</b> $%{y:.2f}<extra></extra>'
    ))

    # Predicted Price on test set
    fig.add_trace(go.Scatter(
        x=x_hist, y=y_pred_flat,
        mode='lines', name='Predicted (Test)',
        line=dict(color='#ff6b6b', width=2, dash='dot'),
        hovertemplate='<b>Predicted:</b> $%{y:.2f}<extra></extra>'
    ))

    # Confidence band — upper (invisible fill anchor)
    fig.add_trace(go.Scatter(
        x=x_future, y=upper_band,
        mode='lines', line=dict(width=0),
        showlegend=False, hoverinfo='skip'
    ))

    # Confidence band — lower with fill to upper
    fig.add_trace(go.Scatter(
        x=x_future, y=lower_band,
        mode='lines', fill='tonexty',
        fillcolor='rgba(246, 211, 101, 0.12)',
        line=dict(width=0),
        name='95% Confidence Band',
        hoverinfo='skip'
    ))

    # Connector dashed line from last actual → first forecast
    fig.add_trace(go.Scatter(
        x=x_connect, y=y_connect,
        mode='lines',
        line=dict(color='rgba(246,211,101,0.5)', width=1.5, dash='dot'),
        showlegend=False, hoverinfo='skip'
    ))

    # Future Forecast main line
    fig.add_trace(go.Scatter(
        x=x_future, y=future_flat,
        mode='lines+markers', name='30-Day Forecast',
        line=dict(color='#f6d365', width=3),
        marker=dict(size=5, color='#f6d365', symbol='circle'),
        hovertemplate='<b>Forecast:</b> $%{y:.2f}<extra></extra>'
    ))

    # Vertical line = forecast start
    fig.add_vline(
        x=forecast_start_x,
        line_dash="dash",
        line_color="rgba(255,255,255,0.35)",
        annotation_text=" Forecast Starts Here",
        annotation_position="top right",
        annotation_font=dict(color='#f6d365', size=12)
    )

    fig.update_layout(
        title=dict(
            text="LSTM+ARO: Price Prediction & 30-Day Future Forecast",
            font=dict(size=22, color='#ffffff')
        ),
        xaxis_title="Date" if last_date is not None else "Time (Days)",
        yaxis_title="Stock Price (USD)",
        template="plotly_dark",
        hovermode="x unified",
        height=560,
        margin=dict(l=20, r=20, t=70, b=20),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.08)'),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.08)', tickprefix='$')
    )
    return fig


def plot_normalized_comparison(df_dict):
    """
    Plots a normalized comparison chart (Percentage Growth) for multiple stocks.
    df_dict: dict of {ticker: dataframe}
    """
    fig = go.Figure()
    
    colors = ['#00f2fe', '#f6d365', '#ff0844', '#a78bfa', '#22c55e', '#f97316']
    
    for i, (ticker, df) in enumerate(df_dict.items()):
        if df is None or df.empty or 'Close' not in df.columns:
            continue
            
        # Normalize to percentage growth from the first available price
        first_price = float(df['Close'].dropna().iloc[0])
        pct_growth = ((df['Close'] - first_price) / first_price) * 100
        
        x_axis = df.index if 'Date' not in df.columns else df['Date']
        
        fig.add_trace(go.Scatter(
            x=x_axis, y=pct_growth,
            mode='lines',
            name=ticker,
            line=dict(color=colors[i % len(colors)], width=2),
            hovertemplate=f'<b>{ticker}</b><br>Growth: %{{y:.2f}}%<extra></extra>'
        ))
        
    fig.update_layout(
        title="Normalized Percentage Growth Comparison",
        xaxis_title="Date",
        yaxis_title="Growth (%)",
        template="plotly_dark",
        hovermode="x unified",
        height=500,
        margin=dict(l=20, r=20, t=60, b=20),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.08)'),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.08)', ticksuffix='%')
    )
    return fig


def plot_correlation_heatmap(df_dict):
    """
    Plots a correlation heatmap between the daily returns of multiple stocks.
    df_dict: dict of {ticker: dataframe}
    """
    import pandas as pd
    
    close_prices = {}
    for ticker, df in df_dict.items():
        if df is not None and not df.empty and 'Close' in df.columns:
            # Handle index to ensure alignment if Dates are available
            if 'Date' in df.columns:
                temp_df = df.set_index('Date')
                close_prices[ticker] = temp_df['Close']
            else:
                close_prices[ticker] = df['Close']
                
    if not close_prices:
        return go.Figure()
        
    combined_df = pd.DataFrame(close_prices)
    # Calculate daily returns
    returns_df = combined_df.pct_change().dropna()
    # Calculate correlation matrix
    corr_matrix = returns_df.corr().round(2)
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        colorscale='Viridis',
        text=corr_matrix.values,
        texttemplate="%{text}",
        hoverinfo="z"
    ))
    
    fig.update_layout(
        title="Stock Return Correlation Matrix",
        template="plotly_dark",
        height=500,
        margin=dict(l=40, r=40, t=60, b=40),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(side="bottom"),
        yaxis=dict(autorange="reversed") # Standard for heatmaps
    )
    return fig

