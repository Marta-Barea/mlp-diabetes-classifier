import src.config as config


def test_config_constants_exist():
    assert isinstance(config.SEED, int)
    assert isinstance(config.TEST_SIZE, float)
    assert isinstance(config.CV_FOLDS, int)
    assert isinstance(config.RANDOM_SEARCH_ITER, int)
    assert isinstance(config.VERBOSE, int)
    assert isinstance(config.DROPOUT_RATE, float)


def test_config_paths_are_strings():
    assert isinstance(config.DATA_PATH, str)
    assert isinstance(config.MODEL_OUTPUT_DIR, str)
    assert isinstance(config.CHECKPOINTS_DIR, str)


def test_config_lists_are_loaded():
    assert isinstance(config.UNITS_LIST, list)
    assert isinstance(config.LR_LIST, list)
    assert isinstance(config.EPOCHS_LIST, list)
    assert isinstance(config.BATCH_SIZE_LIST, list)
