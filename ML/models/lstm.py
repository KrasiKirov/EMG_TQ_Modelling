from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dropout, Dense

try:
    from ML.config import N_STEPS, N_FEATURES, LSTM_UNITS
except ImportError:  # supports running this module directly from ML/
    from config import N_STEPS, N_FEATURES, LSTM_UNITS


def build_model(n_steps: int = N_STEPS, n_features: int = N_FEATURES,
                n_units: int = LSTM_UNITS) -> Sequential:
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
        Dropout(0.3),
        LSTM(n_units, activation='tanh', return_sequences=False),
        Dropout(0.3),
        Dense(1),
    ], name='ankle_torque_lstm')
    return model
