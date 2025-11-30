import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Tumour classification project")

    parser.add_argument("--cuda", action="store_true", help="Enable cuda for xgboost")
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train new models or load existing ones, if true respect the train value in the training config",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path("models"),
        help="Directory to save models",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Directory to save results",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=128,
        help="Image size to resize data to.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of epochs to train for.  Only applies to neural networks.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate for training.  Applies for neural networks and gradient boost",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Neural_Aug",
        help="Name of model to train",
    )
    parser.add_argument(
        "--feature",
        type=str,
        choices=["hog", "lbp", "colour", "combined"],
        default="combined",
        help="Feature extraction method for conventional models",
    )

    args = parser.parse_args()

    return args
