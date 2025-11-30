import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern
from sklearn.preprocessing import LabelEncoder, StandardScaler


def extract_hog_features(images):
    """Extract HOG features from images"""
    print("Extracting HOG features...")
    features = []
    for img in images:
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # Extract HOG features
        fd = hog(
            gray,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            visualize=False,
        )
        features.append(fd)
    return np.array(features)


def extract_lbp_features(images):
    """Extract Local Binary Pattern features"""
    print("Extracting LBP features...")
    features = []
    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # LBP parameters
        radius = 3
        n_points = 8 * radius
        lbp = local_binary_pattern(gray, n_points, radius, method="uniform")
        # Create histogram
        hist, _ = np.histogram(
            lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2)
        )
        # Normalize
        hist = hist.astype("float")
        hist /= hist.sum() + 1e-6
        features.append(hist)
    return np.array(features)


def extract_colour_features(images):
    """Extract colour histogram features"""
    print("Extracting colour histogram features...")
    features = []
    for img in images:
        # Calculate histogram for each channel
        hist_features = []
        for i in range(3):  # RGB channels
            hist = cv2.calcHist([img], [i], None, [32], [0, 256])
            hist = hist.flatten()
            hist = hist / (hist.sum() + 1e-6)  # Normalize
            hist_features.extend(hist)
        features.append(hist_features)
    return np.array(features)


def extract_combined_features(images):
    """Combine multiple feature extraction methods"""
    print("Extracting combined features (HOG + LBP + Colour)...")
    hog_features = extract_hog_features(images)
    lbp_features = extract_lbp_features(images)
    colour_features = extract_colour_features(images)

    # Concatenate all features
    combined = np.hstack([hog_features, lbp_features, colour_features])
    print(f"Combined feature shape: {combined.shape}")
    return combined


def prepare_data_conventional(train_data, valid_data, feature_type="combined"):
    # Load images
    X_train, y_train = train_data
    X_val, y_val = valid_data

    # Extract features
    if feature_type == "hog":
        X_train = extract_hog_features(X_train)
        X_val = extract_hog_features(X_val)
    elif feature_type == "lbp":
        X_train = extract_lbp_features(X_train)
        X_val = extract_lbp_features(X_val)
    elif feature_type == "colour":
        X_train = extract_colour_features(X_train)
        X_val = extract_colour_features(X_val)
    elif feature_type == "combined":
        X_train = extract_combined_features(X_train)
        X_val = extract_combined_features(X_val)
    else:
        raise ValueError(f"Unknown feature type: {feature_type}")

    # Encode labels
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)
    y_val = label_encoder.transform(y_val)

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    print("\nData split:")
    print(f"  Training set: {np.shape(X_train)}")
    print(f"  Validation set: {np.shape(X_val)}")

    return X_train, X_val, y_train, y_val, label_encoder, scaler


def prepare_data_neural(train_data, valid_data):
    # Load images
    X_train, y_train = train_data
    X_val, y_val = valid_data

    # Encode labels
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)
    y_val = label_encoder.transform(y_val)

    print("\nData split:")
    print(f"  Training set: {np.shape(X_train)}")
    print(f"  Validation set: {np.shape(X_val)}")

    return X_train, X_val, y_train, y_val, label_encoder
