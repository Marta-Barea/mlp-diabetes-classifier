import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout


def build_model(input_dim: int, units: int = 12, learning_rate: float = 0.001, dropout_rate: float = 0.1):
    model = Sequential()
    model.add(Dense(units, activation="relu", input_dim=input_dim,
              kernel_initializer="he_uniform"))
    model.add(Dropout(dropout_rate))
    model.add(Dense(8, activation="relu", kernel_initializer="he_uniform"))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation="sigmoid",
              kernel_initializer="glorot_uniform"))

    model.compile(
        loss="binary_crossentropy",
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        metrics=["accuracy"]
    )

    return model
