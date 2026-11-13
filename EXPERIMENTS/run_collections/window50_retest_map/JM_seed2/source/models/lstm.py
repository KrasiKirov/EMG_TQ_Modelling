from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dropout, Dense


def build_model(n_steps: int = 20, n_features: int = 5,
                n_units: int = 8, dropout: float = 0.3) -> Sequential:
    """Two-layer stacked LSTM for ankle torque regression (Nm).

    Architecture
    ------------
    Input  : [batch, n_steps, n_features]
    LSTM 1 : n_units, tanh, return_sequences=True
    Dropout: 30%
    LSTM 2 : n_units, tanh, return_sequences=False
    Dropout: 30%
    Dense  : 1 unit  → predicted ankle torque (Nm)
    """
    model = Sequential([
        Input(shape=(n_steps, n_features)),
        LSTM(n_units, activation='tanh', return_sequences=True),
        Dropout(dropout),
        LSTM(n_units, activation='tanh', return_sequences=False),
        Dropout(dropout),
        Dense(1),
    ], name='ankle_torque_lstm')
    return model
