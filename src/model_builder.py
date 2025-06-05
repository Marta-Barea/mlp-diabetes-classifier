import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


def build_model(input_dim: int, units: int = 12, learning_rate: float = 0.001):
    model = Sequential()
    model.add(Dense(units, activation="relu", input_dim=input_dim))
    model.add(Dense(8, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))

    model.compile(
        loss="binary_crossentropy",
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        metrics=["accuracy"]
    )
    return model
