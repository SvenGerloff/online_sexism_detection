import os
import json
import pandas as pd
import yaml
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import f1_score, balanced_accuracy_score, accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from utils.load_data import load_processed_data
from datetime import datetime
import sys

# Import functions from preprocessing module
sys.path.append('..')

# Load configuration from YAML
config_path = os.getenv("CONFIG_PATH", "../config.yaml")
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

MODEL_TYPE = config['svm_params']['model_type'] 
VECTOR_TYPE = config['svm_params']['feature_extraction']  # "tfidf"
TRAIN_MODEL = config['svm_params'].get("train", False)
TFIDF_PARAMS = config['svm_params'].get('tfidf_params', {})
MODEL_FOLDER = config['paths']['svm_model_dir']
CURRENT_DATETIME = datetime.now().strftime("%Y%m%d_%H%M%S")
MODEL_TIMESTAMP = config["svm_params"].get("model_timestamp")

print("********** SVM *******")

def train_model(X_train, y_train, vectorizer=None):
    if VECTOR_TYPE == "tfidf":
        X_train_transformed = vectorizer.fit_transform(X_train)

    # Apply SMOTE to handle class imbalance
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train_transformed, y_train)

    model = SVC(random_state=42, class_weight='balanced', probability=True)
    model.fit(X_resampled, y_resampled)

    os.makedirs(MODEL_FOLDER, exist_ok=True)

    model_file = os.path.join(MODEL_FOLDER, f"{CURRENT_DATETIME}_svm_model.pkl")
    with open(model_file, "wb") as f:
        pickle.dump((model, vectorizer), f)
    print(f"Model trained and saved as {model_file}")

    return model, vectorizer

def load_model():
    if MODEL_TIMESTAMP:
        model_file = f"{MODEL_TIMESTAMP}_svm_model.pkl"
        model_path = os.path.join(MODEL_FOLDER, model_file)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file with timestamp {MODEL_TIMESTAMP} not found: {model_path}")

        with open(model_path, "rb") as f:
            print(f"Loading model: {model_path}")
            return pickle.load(f)

    raise ValueError("MODEL_TIMESTAMP must be set in config to load a specific model.")

def evaluate_model(model, vectorizer, X_data, y_data, dataset_name):
    if VECTOR_TYPE == "tfidf":
        X_transformed = vectorizer.transform(X_data)

    y_pred = model.predict(X_transformed)
    metrics = {
        "f1": f1_score(y_data, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_data, y_pred),
        "accuracy": accuracy_score(y_data, y_pred),
        "classification_report": classification_report(y_data, y_pred),
        "confusion_matrix": confusion_matrix(y_data, y_pred).tolist(),
    }

    print(f"{dataset_name} Metrics: ")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print("\nClassification Report:\n")
    print(metrics["classification_report"])
    print("\nConfusion Matrix:")
    for row in metrics["confusion_matrix"]:
        print(row)

    return metrics

# Load data
df = load_processed_data()
train_data = df["train"]
test_data = df["test"]
dev_data = df["dev"]

X_train, y_train = train_data["lemma"], train_data["label"]
X_test, y_test = test_data["lemma"], test_data["label"]
X_dev, y_dev = dev_data["lemma"], dev_data["label"]

if VECTOR_TYPE == "tfidf":
    vectorizer = TfidfVectorizer(
        max_features=TFIDF_PARAMS.get('max_features'),
        ngram_range=tuple(TFIDF_PARAMS.get('ngram_range', (1, 1))),
        min_df=TFIDF_PARAMS.get('min_df', 1)
    )

if TRAIN_MODEL:
    print("Training the model")
    model, transformer = train_model(X_train, y_train, vectorizer=vectorizer)
    print("Model training complete.")
else:
    model, transformer = load_model()
    print("Model loaded successfully.")

print("\nEvaluating model on all datasets:")
print("\n##### Train Dataset #####")
evaluate_model(model, transformer, X_train, y_train, "Train")
print("\n##### Dev Dataset #####")
evaluate_model(model, transformer, X_dev, y_dev, "Dev")
print("\n##### Test Dataset #####")
evaluate_model(model, transformer, X_test, y_test, "Test")

print("******************************")