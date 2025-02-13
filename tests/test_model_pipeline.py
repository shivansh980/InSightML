import pytest
import pickle
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import xgboost as xgb
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense # type: ignore
from src.model_loader import ModelLoaderFactory
from src.shap_explainer import ShapExplainerContext
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# ------------------------- Fixtures for Data and Models ------------------------- #

@pytest.fixture(scope='module')
def iris_data():
    """Load Iris dataset."""
    data = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


# RandomForest (Pickle)
@pytest.fixture(scope='module')
def create_pickle_model(iris_data, tmpdir_factory):
    """Train and save a RandomForest model as a .pkl file."""
    X_train, _, y_train, _ = iris_data
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    file_path = tmpdir_factory.mktemp("models").join("random_forest_model.pkl")
    with open(file_path, 'wb') as f:
        pickle.dump(model, f)
    return str(file_path)


# Keras Sequential
@pytest.fixture(scope='module')
def create_keras_model(iris_data, tmpdir_factory):
    """Train and save a Keras Sequential model as a .h5 file."""
    X_train, _, y_train, _ = iris_data
    model = Sequential([
        Dense(10, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(3, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=5, verbose=0)
    file_path = tmpdir_factory.mktemp("models").join("sequential_model.h5")
    model.save(file_path)
    return str(file_path)


# ------------------------- Tests for Model Loading ------------------------- #

def test_pickle_model_loading(create_pickle_model, iris_data):
    loader = ModelLoaderFactory.get_model_loader(create_pickle_model)
    model = loader.load_model(create_pickle_model)
    _, X_test, _, _ = iris_data
    predictions = model.predict(X_test)
    assert len(predictions) == len(X_test)


def test_keras_model_loading(create_keras_model, iris_data):
    loader = ModelLoaderFactory.get_model_loader(create_keras_model)
    model = loader.load_model(create_keras_model)
    _, X_test, _, _ = iris_data
    predictions = model.predict(X_test)
    assert predictions.shape[0] == X_test.shape[0]


# ------------------------- Tests for SHAP Explanations ------------------------- #

@pytest.mark.parametrize("model_type", [
    'RandomForestClassifier', 'Sequential', 'LogisticRegression',
    'XGBoost', 'SVM'
])
def test_shap_explanation(model_type, iris_data):
    X_train, X_test, y_train, _ = iris_data

    # Model Training
    if model_type == 'RandomForestClassifier':
        model = RandomForestClassifier(random_state=42).fit(X_train, y_train)
    elif model_type == 'Sequential':
        model = Sequential([
            Dense(10, activation='relu', input_shape=(X_train.shape[1],)),
            Dense(3, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=5, verbose=0)
    elif model_type == 'LogisticRegression':
        model = LogisticRegression().fit(X_train, y_train)
    elif model_type == 'XGBoost':
        model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss').fit(X_train, y_train)
    elif model_type == 'SVM':
        model = SVC(probability=True).fit(X_train, y_train)
    

    # SHAP Explanation
    explainer_context = ShapExplainerContext(model)
    shap_values, _ = explainer_context.explain_model(model, X_test[:10])

    # Assertions
    assert shap_values is not None, "SHAP values should not be None."
    assert isinstance(shap_values, (list, np.ndarray)), "SHAP values should be a list or ndarray."
