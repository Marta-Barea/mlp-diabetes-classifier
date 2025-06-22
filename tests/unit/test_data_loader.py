import numpy as np
import pytest
from src import data_loader


def test_load_data_shapes():
    X_train, X_test, y_train, y_test = data_loader.load_data()

    assert X_train.shape[1] == X_test.shape[1]
    assert X_train.shape[0] == y_train.shape[0]
    assert X_test.shape[0] == y_test.shape[0]


def test_data_types():
    X_train, X_test, y_train, y_test = data_loader.load_data()

    for arr in [X_train, X_test, y_train, y_test]:
        assert isinstance(arr, np.ndarray)


def test_data_values_are_numeric():
    X_train, X_test, y_train, y_test = data_loader.load_data()

    for arr in [X_train, X_test, y_train, y_test]:
        assert np.isfinite(arr).all()


def test_load_data_file_not_found():
    with pytest.raises(OSError):
        data_loader.load_data(path="nonexistent_file.csv")
