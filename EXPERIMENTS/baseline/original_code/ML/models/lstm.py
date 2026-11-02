from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dropout, Dense


def build_model(n_steps: int = 20, n_features: int = 5) -> Sequential:
    """Two-layer stacked LSTM for ankle torque regression (Nm).

    Architecture
    ------------
    Input  : [batch, 20, 5]
    LSTM 1 : 8 units, tanh, return_sequences=True
    Dropout: 30%
    LSTM 2 : 8 units, tanh, return_sequences=False
    Dropout: 30%
    Dense  : 1 unit  → predicted ankle torque (Nm)
    """
    model = Sequential([
        Input(shape=(n_steps, n_features)),
        LSTM(8, activation='tanh', return_sequences=True),
        Dropout(0.3),
        LSTM(8, activation='tanh', return_sequences=False),
        Dropout(0.3),
        Dense(1),
    ], name='ankle_torque_lstm')
    return model
