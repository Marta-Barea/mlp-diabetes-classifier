import os
import random
import numpy as np
import tensorflow as tf

from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import RandomizedSearchCV
import joblib

from .config import (
    SEED,
    UNITS_LIST,
    LR_LIST,
    EPOCHS_LIST,
    BATCH_SIZE_LIST,
    RANDOM_SEARCH_ITER,
    CV_FOLDS,
    VERBOSE,
    MODEL_OUTPUT_DIR
)
from .data_loader import load_data
from .model_builder import build_model


def trainer():
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    X_train, X_test, y_train, y_test = load_data()

    input_dim = X_train.shape[1]

    default_units = UNITS_LIST[0] if UNITS_LIST else 12
    default_lr = LR_LIST[0] if LR_LIST else 0.001

    mlp = KerasClassifier(
        model=build_model,
        input_dim=input_dim,
        units=default_units,
        learning_rate=default_lr,
        verbose=0
    )

    param_dist = {
        "model__units": UNITS_LIST,
        "model__learning_rate": LR_LIST,
        "epochs": EPOCHS_LIST,
        "batch_size": BATCH_SIZE_LIST
    }

    random_search = RandomizedSearchCV(
        estimator=mlp,
        param_distributions=param_dist,
        n_iter=RANDOM_SEARCH_ITER,
        cv=CV_FOLDS,
        verbose=VERBOSE,
        random_state=SEED
    )

    print("⏳ Starting RandomizedSearchCV...")
    random_search.fit(X_train, y_train)

    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

    best_model_path = os.path.join(MODEL_OUTPUT_DIR, "best_mlp.pkl")
    joblib.dump(random_search.best_estimator_, best_model_path)

    best_params_path = os.path.join(MODEL_OUTPUT_DIR, "best_params.txt")
    with open(best_params_path, "w") as f:
        f.write(str(random_search.best_params_))

    print("\n✅ Training completed.")
    print(f"   • Best model saved at: {best_model_path}")
    print(f"   • Best parameters saved at: {best_params_path}")
    print("\n🔍 Best parameters found:")
    for key, value in random_search.best_params_.items():
        print(f"     • {key}: {value}")


if __name__ == "__main__":
    trainer()
