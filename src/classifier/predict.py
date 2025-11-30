import argparse
from pathlib import Path

import cv2
import joblib
import numpy as np

from classifier.feature_extraction import (
    extract_colour_features,
    extract_combined_features,
    extract_hog_features,
    extract_lbp_features,
)
from classifier.neural_nets import make_neural_net_generator


def predict_conventional(
    image_path,
    model,
    label_encoder,
    scaler,
    feature_type,
    image_shape=(128, 128),
):
    # Load and preprocess image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Predict image is none")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, image_shape)

    img_array = np.array([img])

    # Extract features
    if feature_type == "hog":
        features = extract_hog_features(img_array)
    elif feature_type == "lbp":
        features = extract_lbp_features(img_array)
    elif feature_type == "colour":
        features = extract_colour_features(img_array)
    elif feature_type == "combined":
        features = extract_combined_features(img_array)
    else:
        raise ValueError(f"Unknown feature type: {feature_type}")

    features = scaler.transform(features)

    # Predict
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]

    predicted_class = label_encoder.inverse_transform([prediction])[0]

    print(f"\nPrediction for {image_path}:")
    print(f"  Class: {predicted_class}")
    print(f"  Confidence: {probabilities[prediction]:.4f}")
    print("\nAll probabilities:")
    for i, class_name in enumerate(label_encoder.classes_):
        print(f"  {class_name}: {probabilities[i]:.4f}")


def predict_neural(
    image_path,
    model,
    label_encoder,
    image_shape=(128, 128),
):
    # Load and preprocess image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Predict image is none")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, image_shape)

    img = np.expand_dims(img, axis=0)

    # Predict
    prediction = model.predict(img)[0]
    probabilities = model.predict_proba(img)[0]

    predicted_class = label_encoder.inverse_transform([prediction])[0]

    print(f"\nPrediction for {image_path}:")
    print(f"  Class: {predicted_class}")
    print(f"  Confidence: {probabilities[prediction]:.4f}")
    print("\nAll probabilities:")
    for i, class_name in enumerate(label_encoder.classes_):
        print(f"  {class_name}: {probabilities[i]:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Tumour classification project")

    parser.add_argument("--image-path", type=Path, help="Path for image to predict on")
    parser.add_argument("--model", type=str, help="Path for model to predict with")
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path("models"),
        help="Directory to save models",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Needed for loading model from correct path",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        help="dimension to resize images to.  Images resized to square.",
        default=128,
    )
    args = parser.parse_args()

    # Conventional models train on extracted features instead of images
    conventional_model_names = [
        "Random_Forest",
        "Gradient_Boosting",
        "SVM_RBF",
        "SVM_Linear",
        "K_Nearest",
    ]

    model_name = args.model
    random_state = args.random_state
    image_size = args.image_size
    image_shape = (image_size, image_size)

    models_dir: Path = (
        Path(f"random_state_{random_state}")
        / Path(f"image_size_{image_size}")
        / args.models_dir
    )

    match model_name:
        case "Neural":
            model = make_neural_net_generator(
                (*image_shape, 3), se=False, use_augmentation=False
            )()
            model_path = models_dir / f"{model_name}.weights.h5"
            model.load_weights(model_path)

            label_encoder_path = models_dir / "neural_label_encoder.pkl"
            label_encoder = joblib.load(label_encoder_path)

            predict_neural(
                args.image_path,
                model,
                label_encoder,
                image_shape,
            )

        case "Neural_Aug":
            model = make_neural_net_generator(
                (*image_shape, 3), se=False, use_augmentation=True
            )()
            model_path = models_dir / f"{model_name}.weights.h5"
            model.load_weights(model_path)

            label_encoder_path = models_dir / "neural_label_encoder.pkl"
            label_encoder = joblib.load(label_encoder_path)

            predict_neural(
                args.image_path,
                model,
                label_encoder,
                image_shape,
            )

        case "NeuralSE":
            model = make_neural_net_generator(
                (*image_shape, 3), se=True, use_augmentation=False
            )()
            model_path = models_dir / f"{model_name}.weights.h5"
            model.load_weights(model_path)

            label_encoder_path = models_dir / "neural_label_encoder.pkl"
            label_encoder = joblib.load(label_encoder_path)

            predict_neural(
                args.image_path,
                model,
                label_encoder,
                image_shape,
            )

        case "NeuralSE_Aug":
            model = make_neural_net_generator(
                (*image_shape, 3), se=True, use_augmentation=True
            )()
            model_path = models_dir / f"{model_name}.weights.h5"
            model.load_weights(model_path)

            label_encoder_path = models_dir / "neural_label_encoder.pkl"
            label_encoder = joblib.load(label_encoder_path)

            predict_neural(
                args.image_path,
                model,
                label_encoder,
                image_shape,
            )

        case name if model_name in conventional_model_names:
            model_path = models_dir / f"{name}.pkl"
            model_data = joblib.load(model_path)
            model = model_data["model"]
            label_encoder = model_data["label_encoder"]
            scaler = model_data["scaler"]
            feature_type = model_data["feature_type"]
            predict_conventional(
                args.image_path,
                model,
                label_encoder,
                scaler,
                feature_type,
                image_shape,
            )

        case _:
            raise ValueError("Invalid model name")


if __name__ == "__main__":
    main()
