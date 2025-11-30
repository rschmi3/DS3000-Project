# ruff: noqa: E402
import os
import random
import time
from pathlib import Path
from typing import Any

from classifier.conventional_models import (
    make_gradient_boost_generator,
    make_K_nearest_generator,
    make_random_forest_generator,
    make_svc_generator,
)
from classifier.utils import parse_args

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
# For GPU determinism (makes things slower but reproducible)
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

import joblib
import numpy as np
import tensorflow as tf

from classifier.evaluation import (
    determine_best_model,
    evaluate_model,
    plot_neural_network_activations,
    plot_results,
)
from classifier.feature_extraction import prepare_data_conventional, prepare_data_neural
from classifier.neural_nets import make_neural_net_generator
from classifier.setup_data import download_dataset, load_images_and_labels

tf.config.experimental.enable_op_determinism()

# Set TensorFlow to only allocate what it needs
gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


def make_conventional_train_func(X, y, model):
    def train_func():
        # Train conventional model
        train_start = time.time()
        model.fit(X, y)
        train_end = time.time()
        train_duration = train_end - train_start

        return train_duration

    return train_func


def make_neural_train_func(X, y, epochs, lr, model):
    def train_func():
        # Train neural network
        train_start = time.time()
        model.compile()
        model.fit(
            X,
            y,
            epochs=epochs,
            lr=lr,
        )
        train_end = time.time()
        train_duration = train_end - train_start

        return train_duration

    return train_func


def make_eval_func(X, y, model):
    def eval_func():
        evaluation_start = time.time()
        results = evaluate_model(model, X, y)
        evaluation_end = time.time()
        results["eval_time"] = evaluation_end - evaluation_start
        return results

    return eval_func


def main():
    """Main execution function"""
    args = parse_args()

    print("=" * 60)
    print(f"BRAIN TUMOR CLASSIFICATION {'(CUDA)' if args.cuda else ''}")
    print("=" * 60)

    # Set random seed for reproducibility
    random_state = args.random_state
    random.seed(random_state)
    np.random.seed(random_state)
    tf.random.set_seed(random_state)

    image_size = args.image_size

    # Create model and results directories for random_state and image_size configuration
    models_dir: Path = (
        Path(f"random_state_{random_state}")
        / Path(f"image_size_{image_size}")
        / args.models_dir
    )
    results_dir = (
        Path(f"random_state_{random_state}")
        / Path(f"image_size_{image_size}")
        / args.results_dir
    )

    models_dir.mkdir(exist_ok=True, parents=True)
    results_dir.mkdir(exist_ok=True, parents=True)

    # Load previous results if they exist
    results_path = results_dir / "evaluation_results.pkl"
    if results_path.exists():
        results = joblib.load(results_path)
    else:
        results = {}

    # Download dataset and load data
    dataset_path = download_dataset("deeppythonist/brain-tumor-mri-dataset")
    train_data, valid_data, classes = load_images_and_labels(
        dataset_path=dataset_path, image_size=(image_size, image_size)
    )

    # Extract shape and assert it's a valid tuple
    shape = train_data[0].shape[1:]
    assert len(shape) >= 2, "Image array must have at least 2 dimensions"
    image_shape = shape

    # Conventional models train on extracted features instead of images
    conventional_model_names = [
        "Random_Forest",
        "Gradient_Boosting",
        "SVM_RBF",
        "SVM_Linear",
        "K_Nearest",
    ]

    # Neural networks train directly on images
    neural_model_names = ["Neural_Aug", "Neural", "NeuralSE_Aug", "NeuralSE"]

    # Setup model generator functions for each model type
    model_generators: dict[str, Any] = {
        "Neural": make_neural_net_generator(
            image_shape, se=False, use_augmentation=False
        ),
        "Neural_Aug": make_neural_net_generator(
            image_shape, se=False, use_augmentation=True
        ),
        "NeuralSE": make_neural_net_generator(
            image_shape, se=True, use_augmentation=False
        ),
        "NeuralSE_Aug": make_neural_net_generator(
            image_shape, se=True, use_augmentation=True
        ),
        "Random_Forest": make_random_forest_generator(
            random_state=random_state,
        ),
        "Gradient_Boosting": make_gradient_boost_generator(
            n_estimators=200,
            max_depth=5,
            lr=args.lr,
            cuda=args.cuda,
            random_state=random_state,
        ),
        "SVM_RBF": make_svc_generator(
            kernel="rbf",
            C=10,
            gamma="scale",
            probability=True,
            random_state=random_state,
        ),
        "SVM_Linear": make_svc_generator(
            kernel="linear",
            C=1,
            probability=True,
            random_state=random_state,
        ),
        "K_Nearest": make_K_nearest_generator(),
    }

    model_name = args.model
    trained_models = {}
    train_funcs = {}
    eval_funcs = {}

    match model_name:
        case name if model_name in neural_model_names:
            # Extract features
            X_train, X_val, y_train, y_val, label_encoder = prepare_data_neural(
                train_data,
                valid_data,
            )
            model = model_generators[name]()

            if args.train:
                # Neural networks train on images directly
                train_func = make_neural_train_func(
                    X_train, y_train, args.epochs, args.lr, model
                )
                train_funcs[name] = train_func

                model_data = {"model": model, "label_encoder": label_encoder}

                # Store model
                trained_models[name] = model_data

            else:
                model_path = models_dir / f"{name}.weights.h5"
                model.load_weights(model_path)

            eval_funcs[name] = make_eval_func(X_val, y_val, model)

        case "conventional":
            # Extract features
            X_train, X_val, y_train, y_val, label_encoder, scaler = (
                prepare_data_conventional(
                    train_data, valid_data, feature_type=args.feature
                )
            )
            # Train all conventional_models
            for name in conventional_model_names:
                if args.train:
                    model = model_generators[name]()
                    train_func = make_conventional_train_func(X_train, y_train, model)
                    train_funcs[name] = train_func

                    model_data = {
                        "feature_type": args.feature,
                        "label_encoder": label_encoder,
                        "model": model,
                        "scaler": scaler,
                    }
                    # Store model
                    trained_models[name] = model_data

                else:
                    model_path = models_dir / f"{name}.pkl"
                    model_data = joblib.load(model_path)
                    model = model_data["model"]

                eval_funcs[name] = make_eval_func(X_val, y_val, model)

        case name if model_name in conventional_model_names:
            # Extract features
            X_train, X_val, y_train, y_val, label_encoder, scaler = (
                prepare_data_conventional(
                    train_data, valid_data, feature_type=args.feature
                )
            )
            # Train specific conventional model
            if args.train:
                model = model_generators[name]()
                train_funcs[name] = make_conventional_train_func(
                    X_train, y_train, model
                )

                model_data = {
                    "feature_type": args.feature,
                    "label_encoder": label_encoder,
                    "model": model,
                    "scaler": scaler,
                }
                # Store model
                trained_models[name] = model_data

            else:
                model_path = models_dir / f"{name}.pkl"
                model_data = joblib.load(model_path)
                model = model_data["model"]

            eval_funcs[name] = make_eval_func(X_val, y_val, model)

        case _:
            raise ValueError("Invalid model name")

    print("\n" + "=" * 60)
    print("TRAINING MODELS ON TRAIN SET")
    print("=" * 60)

    for name, train_func in train_funcs.items():
        results[name] = {}
        print(f"\nTraining {name}...")
        train_duration = train_func()
        results[name]["training_time"] = train_duration
        print(f"Training {name} finished: {train_duration}")

    print("\n" + "=" * 60)
    print("EVALUATING MODELS ON VALIDATION SET")
    print("=" * 60)

    for name, eval_func in eval_funcs.items():
        print(f"\nEvaluating {name}")
        results[name] = eval_func()
        print(f"Evaluating {name} finished: {results[name]['eval_time']}")

        # If neural network then run create activation visualizations
        if name in neural_model_names:
            neural_visualizations_dir = results_dir / model_name

            neural_visualizations_dir.mkdir(exist_ok=True, parents=True)

            plot_neural_network_activations(
                valid_data[0],
                valid_data[1],
                classes,
                3,
                trained_models[name]["model"],
                neural_visualizations_dir,
            )

    plot_results(results, classes, results_dir)

    determine_best_model(results)

    if args.train:
        for name, model_data in trained_models.items():
            if name in conventional_model_names:
                model_path = models_dir / f"{name}.pkl"
                print(f"\nSaving model: {name} to {model_path}")
                joblib.dump(model_data, model_path)

            else:
                model_path = models_dir / f"{name}.weights.h5"
                label_encoder_path = models_dir / "neural_label_encoder.pkl"
                model = model_data["model"]
                label_encoder = model_data["label_encoder"]
                model.save_weights(model_path)
                joblib.dump(label_encoder, label_encoder_path)

    evaluation_resuls_path = results_dir / "evaluation_results.pkl"
    joblib.dump(results, evaluation_resuls_path)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"\nResults saved to {args.results_dir} directory")


if __name__ == "__main__":
    main()
