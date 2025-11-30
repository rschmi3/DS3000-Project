import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize


def evaluate_model(model, X_val, y_val):
    # Predict
    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)

    # Accuracy
    accuracy = accuracy_score(y_val, y_pred)

    # Confusion matrix
    cm = confusion_matrix(y_val, y_pred)

    # Calculate ROC Curve and AUC-ROC for multi-class
    y_val_bin = label_binarize(y=y_val, classes=np.unique(y_val))
    assert isinstance(y_val_bin, np.ndarray)

    fpr, tpr, _ = roc_curve(y_val_bin.ravel(), y_proba.ravel())

    auc_roc = roc_auc_score(y_val_bin, y_proba, average="macro", multi_class="ovr")

    # Store test results
    results = {}
    results["test_accuracy"] = accuracy
    results["test_predictions"] = y_pred
    results["test_probabilities"] = y_proba
    results["confusion_matrix"] = cm
    results["auc_roc"] = auc_roc
    results["fpr"] = fpr
    results["tpr"] = tpr

    return results


def plot_accuracy_auc(results, results_dir):
    names = list(results.keys())
    accuracies = [results[name]["test_accuracy"] for name in names]
    auc_scores = [results[name]["auc_roc"] for name in names]

    x = np.arange(len(names))
    width = 0.35

    _, ax1 = plt.subplots(figsize=(12, 6))

    ax1.bar(
        x - width / 2,
        accuracies,
        width,
        label="Accuracy",
        alpha=0.8,
        color="blue",
    )

    ax1.set_xlabel("Model")
    ax1.set_ylabel("Score", color="black")
    ax1.tick_params(axis="y", labelcolor="black")

    ax1.bar(x + width / 2, auc_scores, width, label="AUC-ROC", alpha=0.8, color="red")

    plt.title("Model Performance Comparison")
    plt.xticks(x, names, rotation=0, ha="center")

    lines1, labels1 = ax1.get_legend_handles_labels()
    plt.legend(lines1, labels1, loc="upper right")

    plt.ylim(0, 1.0)
    plt.tight_layout()
    model_comparison_path = results_dir / "model_comparison.png"
    plt.savefig(model_comparison_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_confusion_matrices(results, classes, results_dir):
    n_models = len(results)
    _, axes = plt.subplots(3, 3, figsize=(18, 12))
    axes = axes.flatten()

    for idx, (name, result) in enumerate(results.items()):
        if idx >= len(axes):
            break
        cm = result["confusion_matrix"]
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=axes[idx],
            xticklabels=list(classes),
            yticklabels=list(classes),
        )
        axes[idx].set_title(f"{name}\nAccuracy: {result['test_accuracy']:.4f}")
        axes[idx].set_ylabel("True Label")
        axes[idx].set_xlabel("Predicted Label")

    # Hide unused subplots
    for idx in range(n_models, len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()
    confusion_matrix_path = results_dir / "confusion_matrices.png"
    plt.savefig(confusion_matrix_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_roc_curve(results, results_dir):
    roc_curve_path = results_dir / "roc_curve.png"
    plt.figure(figsize=(10, 8))

    for name, result in results.items():
        fpr = result["fpr"]
        tpr = result["tpr"]
        roc_auc = result["auc_roc"]

        plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})", linewidth=2)

    plt.plot(
        [0, 1],
        [0, 1],
        "k--",
        linewidth=1,
    )
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("ROC Curve", fontsize=14, fontweight="bold")
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(roc_curve_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_gradcam_heatmap(X, model, c, save_path):
    img, heatmap_resized, superimposed_img, pred_class, pred_confidence = (
        model.make_gradcam_heatmap(X)
    )

    # Create figure with subplots
    _, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original image
    axes[0].imshow(img.astype("uint8") if img.max() > 1 else img)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    # Heatmap only
    im = axes[1].imshow(heatmap_resized, cmap="jet", vmin=0, vmax=1)
    axes[1].set_title("Grad-CAM Heatmap")
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046)

    # Overlay
    axes[2].imshow(superimposed_img)
    axes[2].set_title(
        f"Grad-CAM Overlay\nClass: {c}, {pred_class}, Conf: {pred_confidence:.3f}"
    )
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved Grad-CAM visualization to {save_path}")


def plot_feature_visualizations(X, model, save_path, maps_per_layer=8):
    feature_maps = model.make_feature_maps(X, maps_per_layer)
    num_layers = len(feature_maps)

    fig, axes = plt.subplots(num_layers, maps_per_layer, figsize=(50, 50))
    fig.suptitle("Feature Maps", fontsize=16)

    for i, layer_maps in enumerate(feature_maps):
        for j, map in enumerate(layer_maps):
            ax = axes[i, j]
            ax.imshow(map, cmap="viridis")
            ax.axis("off")

    plt.tight_layout()
    plt.show()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved feature visualization to {save_path}")


def plot_neural_network_activations(
    X,
    y,
    classes,
    samples_per_class,
    model,
    save_dir,
):
    for c in classes:
        class_dir = save_dir / f"{c}"
        class_dir.mkdir(exist_ok=True)
        class_indices = np.where(y == c)[0]

        for i in range(samples_per_class):
            grad_cam_path = f"{class_dir}/grad_cam_{i}.png"
            feature_map_path = f"{class_dir}/feature_map_{i}.png"
            image_idx = np.random.choice(class_indices)
            plot_gradcam_heatmap(X[image_idx], model, c, grad_cam_path)
            plot_feature_visualizations(X[image_idx], model, feature_map_path)


def plot_results(results, classes, results_dir):
    """Create visualizations of model performance"""

    # 1. Model comparison bar chart
    plot_accuracy_auc(results, results_dir)

    # 2. Confusion matrices
    plot_confusion_matrices(results, classes, results_dir)

    # 3. ROC Curve
    plot_roc_curve(results, results_dir)


def determine_best_model(results):
    best_accuracy_name = max(results.keys(), key=lambda x: results[x]["test_accuracy"])

    print(f"\nBest Accuracy model ({best_accuracy_name})")
    print(f"  Accuracy: {results[best_accuracy_name]['test_accuracy']:.4f}")
    print(f"  AUC-ROC: {results[best_accuracy_name]['auc_roc']:.4f}")

    best_auc_model = max(results.keys(), key=lambda x: results[x]["auc_roc"])

    print(f"\nBest AUC-ROC model ({best_auc_model})")
    print(f"  Accuracy: {results[best_auc_model]['test_accuracy']:.4f}")
    print(f"  AUC-ROC: {results[best_auc_model]['auc_roc']:.4f}")
