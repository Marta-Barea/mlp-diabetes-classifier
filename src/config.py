import os
import yaml

CONFIG_FILE = os.path.join(os.path.dirname(__file__), os.pardir, "config.yaml")


def _load_config():
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


_config = _load_config()

SEED = int(_config.get("seed", 42))
TEST_SIZE = float(_config.get("test_size", 0.3))

CV_FOLDS = int(_config.get("cv_folds", 3))
RANDOM_SEARCH_ITER = int(_config.get("random_search_iter", 10))
VERBOSE = int(_config.get("verbose", 2))

DATA_PATH = _config.get("data_path", "data/diabetes.csv")
MODEL_OUTPUT_DIR = _config.get("model_output_path", "models")
CHECKPOINTS_DIR = _config.get("checkpoints_outputh_path", "checkpoints")

DROPOUT_RATE = float(_config.get("dropout_rate", 0.1))

search_cfg = _config.get("search", {})
UNITS_LIST = search_cfg.get("units_list", [])
LR_LIST = search_cfg.get("lr_list", [])
EPOCHS_LIST = search_cfg.get("epochs_list", [])
BATCH_SIZE_LIST = search_cfg.get("batch_size_list", [])
