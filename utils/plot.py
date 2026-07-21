import plotly.graph_objects as go
import pandas as pd


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
        title=dict(text="📈 Prediction Comparison", font=dict(size=24, color='#ffffff')),
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
    fig = go.Figure()

    # Normalize column names
    df_plot = df.copy()
    df_plot.columns = [col.strip() for col in df_plot.columns]
    df_plot.rename(columns={'Close/Last': 'Close'}, inplace=True)

    cols_lower = [col.lower() for col in df_plot.columns]

    # Proceed only if OHLC columns exist
    if all(x in cols_lower for x in ['open', 'high', 'low', 'close']):
        # Clean $ and commas from price columns using pd.to_numeric for safety
        for c in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if c in df_plot.columns:
                df_plot[c] = pd.to_numeric(
                    df_plot[c].astype(str).str.replace(r'[\$,]', '', regex=True),
                    errors='coerce'
                )

        # Drop rows where price data is NaN
        df_plot.dropna(subset=['Open', 'High', 'Low', 'Close'], inplace=True)

        # Calculate moving averages
        df_plot['SMA20'] = df_plot['Close'].rolling(window=20).mean()
        df_plot['SMA50'] = df_plot['Close'].rolling(window=50).mean()

        x_axis = df_plot['Date'] if 'Date' in df_plot.columns else df_plot.index

        fig.add_trace(go.Candlestick(
            x=x_axis,
            open=df_plot['Open'],
            high=df_plot['High'],
            low=df_plot['Low'],
            close=df_plot['Close'],
            name='Market Data'
        ))

        fig.add_trace(go.Scatter(
            x=x_axis, y=df_plot['SMA20'],
            line=dict(color='orange', width=1.5), name='SMA 20'
        ))
        fig.add_trace(go.Scatter(
            x=x_axis, y=df_plot['SMA50'],
            line=dict(color='cyan', width=1.5), name='SMA 50'
        ))

        fig.update_layout(
            title="📊 Interactive Candlestick Chart with Moving Averages",
            yaxis_title="Stock Price ($)",
            template="plotly_dark",
            xaxis_rangeslider_visible=False,
            hovermode="x unified",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
            yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)')
        )
    return fig


def plot_future_forecast(y_true, y_pred, future_preds):
    fig = go.Figure()

    y_true_flat = y_true.flatten() if hasattr(y_true, 'flatten') else y_true
    y_pred_flat = y_pred.flatten() if hasattr(y_pred, 'flatten') else y_pred
    future_flat = future_preds.flatten() if hasattr(future_preds, 'flatten') else future_preds

    time_true = list(range(len(y_true_flat)))
    time_future = list(range(len(y_true_flat), len(y_true_flat) + len(future_flat)))

    # Actual Price
    fig.add_trace(go.Scatter(
        x=time_true, y=y_true_flat, mode='lines', name='Actual Price',
        line=dict(color='#00f2fe', width=3),
        fill='tozeroy', fillcolor='rgba(0, 242, 254, 0.1)'
    ))

    # Predicted Price
    fig.add_trace(go.Scatter(
        x=time_true, y=y_pred_flat, mode='lines', name='Predicted Price (Test)',
        line=dict(color='#ff0844', width=3, dash='dot')
    ))

    # Future Forecast
    fig.add_trace(go.Scatter(
        x=time_future, y=future_flat, mode='lines', name='Future Forecast (30 Days)',
        line=dict(color='#f6d365', width=3, dash='dash')
    ))

    fig.update_layout(
        title=dict(text="📈 Prediction & Future Forecast", font=dict(size=24, color='#ffffff')),
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
