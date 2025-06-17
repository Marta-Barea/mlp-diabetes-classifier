import sys
import pathlib

from src.train import trainer
from src.evaluate import evaluator


def runner():
    project_root = pathlib.Path(__file__).parent.parent.resolve()
    sys.path.insert(0, str(project_root))

    models_dir = project_root / "models"
    models_dir.mkdir(exist_ok=True)

    print("=== STARTING TRAINING ===")
    try:
        trainer()
    except Exception as e:
        print("❌ Error during training:")
        print(e)
        sys.exit(1)

    print("\n=== STARTING EVALUATION ===")
    try:
        evaluator()
    except Exception as e:
        print("❌ Error during evaluation:")
        print(e)
        sys.exit(1)

    print("\n=== PROCESS COMPLETED ===")


if __name__ == "__main__":
    runner()
