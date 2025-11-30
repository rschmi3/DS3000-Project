import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


def make_random_forest_generator(
    n_estimators=200,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    n_jobs=-1,
    random_state=42,
):
    def model_generator():
        return RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            n_jobs=n_jobs,
            random_state=random_state,
        )

    return model_generator


def make_gradient_boost_generator(
    n_estimators=200, lr=0.1, max_depth=6, cuda=True, random_state=42
):
    def model_generator():
        return xgb.XGBClassifier(
            n_estimators=n_estimators,
            learning_rate=lr,
            max_depth=max_depth,
            random_state=random_state,
            device="cuda" if cuda else "cpu",
        )

    return model_generator


def make_svc_generator(
    kernel="rbf",
    C=10,
    gamma="scale",
    probability=True,
    random_state=42,
):
    def model_generator():
        return SVC(
            kernel=kernel,
            C=C,
            gamma=gamma,
            probability=probability,
            random_state=random_state,
        )

    return model_generator


def make_K_nearest_generator(n_neighbors=5, weights="distance", n_jobs=-1):
    def model_generator():
        return KNeighborsClassifier(
            n_neighbors=n_neighbors, weights=weights, n_jobs=n_jobs
        )

    return model_generator
