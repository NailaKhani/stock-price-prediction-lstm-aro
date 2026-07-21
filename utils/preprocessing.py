import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_csv_data(path, column='Close'):
    df = pd.read_csv(path)
    df = df.dropna()

    print("📋 Original columns:", df.columns.tolist())

    # Clean column names
    df.columns = [col.strip() for col in df.columns]

    # Rename 'Close/Last' to 'Close' for consistent access
    if 'Close/Last' in df.columns:
        df.rename(columns={'Close/Last': 'Close'}, inplace=True)

    print("✅ Columns after rename:", df.columns.tolist())

    # Clean dollar signs and convert to float
    df['Close'] = df['Close'].replace('[\$,]', '', regex=True).astype(float)

    return df[[column]]

def preprocess_data(df, sequence_length=60):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df)

    X, y = [], []
    for i in range(sequence_length, len(scaled)):
        X.append(scaled[i-sequence_length:i])
        y.append(scaled[i])

    return np.array(X), np.array(y), scaler

def forecast_future(model, scaler, last_sequence, n_days=30):
    """
    Forecasts future stock prices.
    last_sequence: shape (1, sequence_length, 1) - scaled data
    """
    future_preds_scaled = []
    current_seq = last_sequence.copy()
    
    for _ in range(n_days):
        pred = model.predict(current_seq, verbose=0)
        future_preds_scaled.append(pred[0, 0])
        # Slide the window
        current_seq = np.append(current_seq[:, 1:, :], [[pred[0]]], axis=1)
        
    future_preds_scaled = np.array(future_preds_scaled).reshape(-1, 1)
    future_preds = scaler.inverse_transform(future_preds_scaled)
    return future_preds
