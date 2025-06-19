import datetime
import random
from pathlib import Path

import numpy as np
import tensorflow as tf
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import RandomizedSearchCV

from .config import (
    SEED, UNITS_LIST, LR_LIST, EPOCHS_LIST, BATCH_SIZE_LIST, RANDOM_SEARCH_ITER,
    CV_FOLDS, VERBOSE, MODEL_OUTPUT_DIR, CHECKPOINTS_DIR, DROPOUT_RATE
)
from .data_loader import load_data
from .model_builder import build_model
from .utils.plot_metrics import plot_metrics


def trainer():
    random.seed(SEED)
    np.random.seed(SEED)
    tf.keras.utils.set_random_seed(SEED)
    tf.config.experimental.enable_op_determinism()

    X_train, _, y_train, _ = load_data()
    input_dim = X_train.shape[1]

    base_mlp = KerasClassifier(
        model=build_model,
        input_dim=input_dim,
        dropout_rate=DROPOUT_RATE
    )

    param_dist = {
        "model__units": UNITS_LIST,
        "model__learning_rate": LR_LIST,
        "epochs": EPOCHS_LIST,
        "batch_size": BATCH_SIZE_LIST
    }

    search = RandomizedSearchCV(
        estimator=base_mlp,
        param_distributions=param_dist,
        n_iter=RANDOM_SEARCH_ITER,
        cv=CV_FOLDS,
        verbose=VERBOSE,
        random_state=SEED,
        n_jobs=-1
    )

    print("⏳ Starting RandomizedSearchCV...")
    search.fit(X_train, y_train)
    best_params = search.best_params_
    print("🔍 Best hyperparameters:", best_params)

    run_ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    ckpt_dir = Path(CHECKPOINTS_DIR)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"best_val_acc_{run_ts}.weights.h5"

    final_model = build_model(
        input_dim=input_dim,
        units=best_params["model__units"],
        learning_rate=best_params["model__learning_rate"],
        dropout_rate=DROPOUT_RATE
    )

    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(ckpt_path),
        monitor="val_accuracy",
        mode="max",
        save_weights_only=True,
        save_best_only=True,
        verbose=0
    )

    print("🚀 Training final model...")
    results = final_model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=best_params["epochs"],
        batch_size=best_params["batch_size"],
        callbacks=[checkpoint_cb],
        verbose=1
    )

    plot_metrics(results)

    final_model.load_weights(ckpt_path)

    out_ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    model_out_dir = Path(MODEL_OUTPUT_DIR)
    model_out_dir.mkdir(parents=True, exist_ok=True)

    final_model_path = model_out_dir / f"best_mlp_{out_ts}.h5"
    final_model.save(final_model_path)

    params_path = model_out_dir / f"best_params_{out_ts}.txt"
    with open(params_path, "w") as f:
        f.write(str(best_params))

    print("✅ Done.")
    print(f"   • Model and weights: {final_model_path}")
    print(f"   • Hyperparameters:   {params_path}")


if __name__ == "__main__":
    trainer()
