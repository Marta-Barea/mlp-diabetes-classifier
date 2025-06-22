from src.model_builder import build_model
from src.data_loader import load_data


def test_model_output_shape():
    X_train, _, _, _ = load_data()
    input_dim = X_train.shape[1]
    model = build_model(input_dim=input_dim)

    assert model.input_shape == (None, input_dim)
    assert model.output_shape == (None, 1)


def test_model_is_compiled():
    X_train, _, _, _ = load_data()
    model = build_model(input_dim=X_train.shape[1])

    assert model.optimizer is not None
    assert model.loss == "binary_crossentropy"


def test_model_has_correct_layers():
    X_train, _, _, _ = load_data()
    model = build_model(input_dim=X_train.shape[1])

    layer_types = [type(layer).__name__ for layer in model.layers]

    assert layer_types.count("Dense") == 3
    assert layer_types.count("Dropout") == 2
    assert layer_types[-1] == "Dense"
    assert model.layers[-1].activation.__name__ == "sigmoid"
