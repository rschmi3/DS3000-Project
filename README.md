# Brain Tumor Classifier

A machine learning project for classifying brain tumors from MRI images using both conventional ML models and deep neural networks.

Classes of images: glioma, meningioma, notumour, and pituitary.

Evaluation results are aggregated with results from previously train models with the same image size and random seed.

Data is automatically pulled from [kaggle](https://www.kaggle.com/datasets/deeppythonist/brain-tumor-mri-dataset).

## Installation

```bash
cd src
pip install -e .
```

## Available Models

**Neural Networks:**
- `Neural` - Basic CNN
- `Neural_Aug` - CNN with data augmentation
- `NeuralSE` - CNN with Squeeze-and-Excitation blocks
- `NeuralSE_Aug` - SE-CNN with augmentation

**Conventional Models:**
- `Random_Forest`
- `Gradient_Boosting`
- `SVM_RBF`
- `SVM_Linear`
- `K_Nearest`

## Training

Train a specific model:
```bash
tumour-classifier --train --model Neural_Aug
```

Train all conventional models:
```bash
tumour-classifier --train --model conventional --feature combined
```

### Training Options

- `--train` - Train new models (omit to load existing)
- `--model` - Model name (default: `Neural_Aug`)
- `--feature` - Feature extraction for conventional models: `hog`, `lbp`, `colour`, or `combined` (default: `combined`)
- `--epochs` - Training epochs for neural networks (default: 20)
- `--lr` - Learning rate (default: 0.001)
- `--image-size` - Input image dimensions (default: 128)
- `--random-state` - Random seed (default: 42)
- `--cuda` - Enable CUDA for XGBoost
- `--models-dir` - Model save directory (default: `models`)
- `--results-dir` - Results save directory (default: `results`)

## Prediction

Predict on a single image:
```bash
tumour-predictor --image-path /path/to/image.jpg --model Neural_Aug
```

### Prediction Options

- `--image-path` - Path to image file
- `--model` - Model name to use
- `--image-size` - Image dimensions (default: 128)
- `--random-state` - Must match training value (default: 42)
- `--models-dir` - Model directory (default: `models`)

## Output

Models and results are organized by random state and image size:
```
random_state_{seed}/
  image_size_{size}/
    models/
    results/
```

Results include confusion matrices, roc curve, model comparison bar graph, and activation visualizations for neural networks.
