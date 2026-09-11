"""Train and evaluate an LSTM model for synthetic energy-consumption data."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential


SEED = 42
SEQUENCE_LENGTH = 24
TEST_FRACTION = 0.2
FEATURES = ["Temperature", "Economic_Index"]
TARGET = "Energy_Consumption"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "hypothetical_energy_data.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"


def create_sequences(
    features: np.ndarray, target: np.ndarray, sequence_length: int
) -> tuple[np.ndarray, np.ndarray]:
    """Create fixed-length feature windows and their following target values."""
    x_values, y_values = [], []
    for index in range(len(features) - sequence_length):
        x_values.append(features[index : index + sequence_length])
        y_values.append(target[index + sequence_length])
    return np.asarray(x_values), np.asarray(y_values)


def build_model(sequence_length: int, feature_count: int) -> Sequential:
    """Build the LSTM regression model."""
    model = Sequential(
        [
            tf.keras.Input(shape=(sequence_length, feature_count)),
            LSTM(50, return_sequences=True),
            Dropout(0.2),
            LSTM(50),
            Dropout(0.2),
            Dense(25, activation="relu"),
            Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mse")
    return model


def main() -> None:
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    data = pd.read_csv(DATA_PATH, parse_dates=["Date"])
    data = data.sort_values("Date").dropna(subset=FEATURES + [TARGET])

    split_row = int(len(data) * (1 - TEST_FRACTION))
    if split_row <= SEQUENCE_LENGTH:
        raise ValueError("The dataset is too small for the configured sequence length.")

    # Fit scalers only on the training period to avoid leaking future information.
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()
    feature_scaler.fit(data.iloc[: split_row][FEATURES])
    target_scaler.fit(data.iloc[: split_row][[TARGET]])

    scaled_features = feature_scaler.transform(data[FEATURES])
    scaled_target = target_scaler.transform(data[[TARGET]]).ravel()
    x_all, y_all = create_sequences(scaled_features, scaled_target, SEQUENCE_LENGTH)
    prediction_dates = data["Date"].iloc[SEQUENCE_LENGTH:].reset_index(drop=True)

    # Sequence index corresponding to the first row in the test period.
    split_sequence = split_row - SEQUENCE_LENGTH
    x_train, x_test = x_all[:split_sequence], x_all[split_sequence:]
    y_train, y_test = y_all[:split_sequence], y_all[split_sequence:]
    test_dates = prediction_dates.iloc[split_sequence:]

    model = build_model(SEQUENCE_LENGTH, len(FEATURES))
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,
    )
    history = model.fit(
        x_train,
        y_train,
        epochs=20,
        batch_size=32,
        validation_split=0.2,
        shuffle=False,
        callbacks=[early_stopping],
        verbose=1,
    )

    predicted_scaled = model.predict(x_test, verbose=0)
    predicted = target_scaler.inverse_transform(predicted_scaled).ravel()
    actual = target_scaler.inverse_transform(y_test.reshape(-1, 1)).ravel()

    rmse = mean_squared_error(actual, predicted) ** 0.5
    mae = mean_absolute_error(actual, predicted)
    print(f"Test period: {test_dates.min().date()} to {test_dates.max().date()}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")

    OUTPUT_DIR.mkdir(exist_ok=True)
    model.save(OUTPUT_DIR / "energy_lstm.keras")

    plt.figure(figsize=(10, 5))
    plt.plot(history.history["loss"], label="Training loss")
    plt.plot(history.history["val_loss"], label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Mean squared error")
    plt.title("LSTM training progress")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "training_loss.png", dpi=150)
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.plot(test_dates, actual, label="Actual")
    plt.plot(test_dates, predicted, "--", label="Predicted")
    plt.xlabel("Date")
    plt.ylabel("Energy consumption")
    plt.title("Actual and predicted energy consumption")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "predictions.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
