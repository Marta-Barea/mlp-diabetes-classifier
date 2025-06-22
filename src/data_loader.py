import numpy as np
from sklearn.model_selection import train_test_split
from .config import DATA_PATH, TEST_SIZE, SEED


def load_data(path=DATA_PATH):

    data = np.loadtxt(path, delimiter=",", skiprows=1)

    X = data[:, :-1]
    y = data[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED
    )
    return X_train, X_test, y_train, y_test
