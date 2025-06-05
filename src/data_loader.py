import numpy as np
from sklearn.model_selection import train_test_split
from .config import DATA_PATH, TEST_SIZE, SEED


def load_data():

    data = np.loadtxt(DATA_PATH, delimiter=",", skiprows=1)

    X = data[:, :8]
    y = data[:, 8]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED
    )
    return X_train, X_test, y_train, y_test
