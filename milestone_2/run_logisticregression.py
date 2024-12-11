import os
import json
import csv
import pandas as pd
import yaml
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, balanced_accuracy_score, accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from gensim.models import Word2Vec
import numpy as np
from datetime import datetime

# Load configuration from YAML
config_path = os.getenv("CONFIG_PATH", "config.yaml")
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

MODEL_TYPE = config['lr_params1']['model_type']  # "tf-idf_logistic_regression", or "word2vec_logistic_regression"(using lr_params2)
TRAIN_MODEL = config['lr_params1'].get("train", False)
VECTOR_TYPE = config['lr_params1'].get("feature_extraction", "tfidf")
TFIDF_PARAMS = config['lr_params1'].get('tfidf_params', {})
SMOTE_PARAMS = config['lr_params1'].get('smote_params', {})
MODEL_FOLDER = config['paths']['lr_model_dir']
CURRENT_DATETIME = datetime.now().strftime("%Y%m%d_%H%M%S")
MODEL_TIMESTAMP = config["lr_params1"].get("model_timestamp")
OUTPUT_FOLDER = config['paths']['output_dir']

print(f"********** Logistic Regression *******")

def train_model(X_train, y_train, vectorizer=None, word2vec_model=None):
    if VECTOR_TYPE == "tfidf":
        X_train_transformed = vectorizer.fit_transform(X_train)
        transformer = vectorizer
    elif VECTOR_TYPE == "word2vec":
        X_train_transformed = vectorize_sentences(X_train, word2vec_model)
        transformer = word2vec_model
    else:
        raise ValueError(f"Invalid vector type: {VECTOR_TYPE}")

    model = LogisticRegression(max_iter=500, class_weight='balanced')

    # Apply SMOTE if specified in the configuration
    if SMOTE_PARAMS:
        smote = SMOTE(random_state=42, **SMOTE_PARAMS)
        X_train_transformed, y_train = smote.fit_resample(X_train_transformed, y_train)

    model.fit(X_train_transformed, y_train)

    os.makedirs(MODEL_FOLDER, exist_ok=True)

    model_file = os.path.join(MODEL_FOLDER, f"{CURRENT_DATETIME}_{MODEL_TYPE}_model.pkl")
    with open(model_file, "wb") as f:
        pickle.dump((model, transformer), f)
    print(f"Model trained and saved as {model_file}")

    return model, transformer

def load_model():
    if MODEL_TIMESTAMP:
        model_file = f"{MODEL_TIMESTAMP}_{MODEL_TYPE}_model.pkl"
        model_path = os.path.join(MODEL_FOLDER, model_file)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file with timestamp {MODEL_TIMESTAMP} not found: {model_path}")

        with open(model_path, "rb") as f:
            print(f"Loading model: {model_path}")
            return pickle.load(f)

    raise ValueError("MODEL_TIMESTAMP must be set in config to load a specific model.")

def evaluate_model(model, transformer, X_data, y_data, dataset_name, file_writer=None):
    if VECTOR_TYPE == "tfidf":
        X_transformed = transformer.transform(X_data)
    elif VECTOR_TYPE == "word2vec":
        X_transformed = vectorize_sentences(X_data, transformer)
    else:
        raise ValueError(f"Invalid vector type: {VECTOR_TYPE}")

    y_pred = model.predict(X_transformed)
    metrics = {
        "f1": f1_score(y_data, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_data, y_pred),
        "accuracy": accuracy_score(y_data, y_pred),
        "classification_report": classification_report(y_data, y_pred),
        "confusion_matrix": confusion_matrix(y_data, y_pred).tolist(),
    }

    if file_writer:
        file_writer.writerow([dataset_name, metrics["f1"], metrics["balanced_accuracy"], metrics["accuracy"]])
    
    print(f"Evaluation metrics for {dataset_name}:")
    print(f"F1 Score: {metrics['f1']}")
    print(f"Balanced Accuracy: {metrics['balanced_accuracy']}")
    print(f"Accuracy: {metrics['accuracy']}")
    print("Classification Report:")
    print(metrics["classification_report"])
    print("Confusion Matrix:")
    print(np.array(metrics["confusion_matrix"]))




if TRAIN_MODEL:
    print("Training the model")
    model, transformer = train_model(X_train, y_train, vectorizer=vectorizer)
    #results_file = os.path.join(MODEL_FOLDER, f"{CURRENT_DATETIME}_naive_bayes_results.txt")
    #with open(results_file, "w") as file_writer:
    #    file_writer.write("******* Training Parameters *******\n")
    #    file_writer.write(f"Feature Extraction: {VECTOR_TYPE}\n")
    #    if VECTOR_TYPE == "tfidf":
    #        file_writer.write(f"TFIDF Parameters: {TFIDF_PARAMS}\n")
    #    file_writer.write(f"Training Samples: {len(X_train)}\n")
    #    file_writer.write("************************************\n")

    #    metrics_train = evaluate_model(model, transformer, X_train, y_train, "Train", file_writer)
    #    metrics_test = evaluate_model(model, transformer, X_test, y_test, "Test", file_writer)
    #    metrics_dev = evaluate_model(model, transformer, X_dev, y_dev, "Dev", file_writer)

    #save_dataset_output(model, transformer, X_train, y_train, "Train")
    save_dataset_output(model, transformer, X_test, y_test, "Test")
    #save_dataset_output(model, transformer, X_dev, y_dev, "Dev")

    #print(f"Model saved saved to {results_file}")
    print("******************************")
else:
    model, transformer = load_model()#
    save_dataset_output(model, transformer, X_test, y_test, "Test")

    print("\nEvaluating model on all datasets:")
    print("\n##### Train Dataset #####")
    evaluate_model(model, transformer, X_train, y_train, "Train", file_writer=None)
    print("\n##### Dev Dataset #####")
    evaluate_model(model, transformer, X_dev, y_dev, "Dev", file_writer=None)
    print("\n##### Test Dataset #####")
    evaluate_model(model, transformer, X_test, y_test, "Test", file_writer=None)

    print("****************************** \n")