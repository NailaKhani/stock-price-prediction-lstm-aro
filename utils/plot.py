import plotly.graph_objects as go

def plot_results(y_true, y_pred):
    fig = go.Figure()
    
    # Handling data shapes
    y_true_flat = y_true.flatten() if hasattr(y_true, 'flatten') else y_true
    y_pred_flat = y_pred.flatten() if hasattr(y_pred, 'flatten') else y_pred
    
    # Actual Price - Glowing Blue
    fig.add_trace(go.Scatter(
        y=y_true_flat, 
        mode='lines', 
        name='Actual Price', 
        line=dict(color='#00f2fe', width=3),
        fill='tozeroy', 
        fillcolor='rgba(0, 242, 254, 0.1)'
    ))
    
    # Predicted Price - Glowing Pink/Red
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
