import plotly.graph_objects as go

def plot_results(y_true, y_pred):
    fig = go.Figure()
    
    # Handling data shapes
    y_true_flat = y_true.flatten() if hasattr(y_true, 'flatten') else y_true
    y_pred_flat = y_pred.flatten() if hasattr(y_pred, 'flatten') else y_pred
    
    fig.add_trace(go.Scatter(y=y_true_flat, mode='lines', name='Actual Price', line=dict(color='#1f77b4', width=2)))
    fig.add_trace(go.Scatter(y=y_pred_flat, mode='lines', name='Predicted Price', line=dict(color='#ff7f0e', width=2)))
    
    fig.update_layout(
        title="📈 Stock Price Prediction (Actual vs Predicted)",
        xaxis_title="Time (Days)",
        yaxis_title="Stock Price",
        template="plotly_dark",
        hovermode="x unified",
        margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig
