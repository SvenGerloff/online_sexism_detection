import os
import json
import csv
import pandas as pd
import yaml
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from utils.load_data import load_processed_data
from sklearn.metrics import f1_score, balanced_accuracy_score, accuracy_score, classification_report, confusion_matrix
from datetime import datetime

# Load configuration from YAML
config_path = os.getenv("CONFIG_PATH", "../config.yaml")
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

VECTOR_TYPE = config['nb_params']['feature_extraction']  # "tfidf" or "word2vec"
TRAIN_MODEL = config['nb_params'].get("train", False)
TFIDF_PARAMS = config['nb_params'].get('tfidf_params', {})
MODEL_FOLDER = config['paths']['nb_model_dir']
CURRENT_DATETIME = datetime.now().strftime("%Y%m%d_%H%M%S")
MODEL_TIMESTAMP = config["nb_params"].get("model_timestamp")

def train_model(X_train, y_train, vectorizer=None, word2vec_model=None):
    if VECTOR_TYPE == "tfidf":
        X_train_transformed = vectorizer.fit_transform(X_train)
        transformer = vectorizer
    elif VECTOR_TYPE == "word2vec":
        transformer = None

    model = MultinomialNB()
    model.fit(X_train_transformed, y_train)

    os.makedirs(MODEL_FOLDER, exist_ok=True)

    model_file = os.path.join(MODEL_FOLDER, f"{CURRENT_DATETIME}_naive_bayes_model.pkl")
    with open(model_file, "wb") as f:
        pickle.dump((model, transformer), f)
    print(f"Model trained and saved as {model_file}")

    return model, transformer

def load_model():
    if MODEL_TIMESTAMP:
        model_file = f"naive_bayes_model_{MODEL_TIMESTAMP}.pkl"
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
        pass

    y_pred = model.predict(X_transformed)
    metrics = {
        "f1": f1_score(y_data, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_data, y_pred),
        "accuracy": accuracy_score(y_data, y_pred),
        "classification_report": classification_report(y_data, y_pred),
        "confusion_matrix": confusion_matrix(y_data, y_pred).tolist(),
    }

    if file_writer:
        file_writer.write("\n")
        file_writer.write(f"******* {dataset_name} Metrics *******\n")
        file_writer.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
        file_writer.write(f"Balanced Accuracy: {metrics['balanced_accuracy']:.4f}\n")
        file_writer.write(f"F1 Score: {metrics['f1']:.4f}\n")
        file_writer.write("\nClassification Report:\n")
        file_writer.write(metrics["classification_report"])
        file_writer.write("\nConfusion Matrix:\n")
        for row in metrics["confusion_matrix"]:
            file_writer.write(f"{row}\n")
        file_writer.write("************************************\n")

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

def save_dataset_output(model, transformer, X_data, y_data, dataset_name):

    if VECTOR_TYPE == "tfidf":
        X_transformed = transformer.transform(X_data)
        feature_names = transformer.get_feature_names_out()
        model_tokens_list = []
        for i in range(X_transformed.shape[0]):
            token_indices = X_transformed[i].nonzero()[1]
            token_list = [feature_names[idx] for idx in token_indices]
            model_tokens_list.append(json.dumps(token_list))
    elif VECTOR_TYPE == "word2vec":
        pass

    y_pred = model.predict(transformer.transform(X_data))

    df_output = pd.DataFrame({
        "lem_tokens": X_data,
        "model_tokens": model_tokens_list,
        "y_pred": y_pred,
        "y_true": y_data
    })

    df_output.to_csv(
        os.path.join(MODEL_FOLDER, f"{CURRENT_DATETIME}_{dataset_name.lower()}_output.csv"),
        index=False,
        encoding='utf-8',
        quoting=csv.QUOTE_MINIMAL,
        quotechar="'"
    )
    print(f"{dataset_name} dataset output saved.")

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
elif VECTOR_TYPE == "word2vec":
    vectorizer = None

if TRAIN_MODEL:
    print("Training the model")
    model, transformer = train_model(X_train, y_train, vectorizer=vectorizer)
    results_file = os.path.join(MODEL_FOLDER, f"{CURRENT_DATETIME}_naive_bayes_results.txt")
    with open(results_file, "w") as file_writer:
        file_writer.write("******* Training Parameters *******\n")
        file_writer.write(f"Feature Extraction: {VECTOR_TYPE}\n")
        if VECTOR_TYPE == "tfidf":
            file_writer.write(f"TFIDF Parameters: {TFIDF_PARAMS}\n")
        file_writer.write(f"Training Samples: {len(X_train)}\n")
        file_writer.write("************************************\n")

        metrics_train = evaluate_model(model, transformer, X_train, y_train, "Train", file_writer)
        metrics_test = evaluate_model(model, transformer, X_test, y_test, "Test", file_writer)
        metrics_dev = evaluate_model(model, transformer, X_dev, y_dev, "Dev", file_writer)

    save_dataset_output(model, transformer, X_train, y_train, "Train")
    save_dataset_output(model, transformer, X_test, y_test, "Test")
    save_dataset_output(model, transformer, X_dev, y_dev, "Dev")

    print(f"Results and evaluation metrics saved to {results_file}")
else:
    print("Loading existing model")
    model, transformer = load_model()

    print("\nEvaluating existing model on all datasets:")
    print("\n##### Train Dataset #####")
    evaluate_model(model, transformer, X_train, y_train, "Train", file_writer=None)
    print("\n##### Test Dataset #####")
    evaluate_model(model, transformer, X_test, y_test, "Test", file_writer=None)
    print("\n##### Dev Dataset #####")
    evaluate_model(model, transformer, X_dev, y_dev, "Dev", file_writer=None)

    save_dataset_output(model, transformer, X_train, y_train, "Train")
    save_dataset_output(model, transformer, X_test, y_test, "Test")
    save_dataset_output(model, transformer, X_dev, y_dev, "Dev")