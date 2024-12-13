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
import sys
# Import functions from preprocessing module
sys.path.append('..')
from utils.load_data import load_processed_data

# Load configuration from YAML
config_path = os.getenv("CONFIG_PATH", "../config.yaml")
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
        smote = SMOTE(random_state=42, **smote_params)
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
    
    print(f"Evaluation metrics for {dataset_name}:")
    print(f"F1 Score: {metrics['f1']}")
    print(f"Balanced Accuracy: {metrics['balanced_accuracy']}")
    print(f"Accuracy: {metrics['accuracy']}")
    print("Classification Report:")
    print(metrics["classification_report"])
    print("Confusion Matrix:")
    print(np.array(metrics["confusion_matrix"]))

def save_dataset_output(model, transformer, X_data, y_data, dataset_name):
    lem_tokens_lists = X_data.apply(to_token_list)

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
    lem_tokens_json = lem_tokens_lists.apply(lambda t: json.dumps(t))

    df_output = pd.DataFrame({
        "lem_tokens": lem_tokens_json,
        "model_tokens": model_tokens_list,
        "label": y_data,
        "y_pred": y_pred,
        "prob_0":  predicted_probs[:, 0],
        "prob_1": predicted_probs[:, 1],
    })

    df_output = df_output.applymap(remove_surrogates)
    file_output = os.path.join(OUTPUT_FOLDER, f"lr_prediction.csv")

    df_output.to_csv(
        os.path.join(MODEL_FOLDER, f"{dataset_name}_output_{CURRENT_DATETIME}.csv"),
        index=False,
        encoding='utf-8',
        sep=',',
        quoting=csv.QUOTE_MINIMAL,
        quotechar="'",
        escapechar='\\'
    )
    print(f"{dataset_name} prediction saved to {file_output}")

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
    #results_file = os.path.join(MODEL_FOLDER, f"{CURRENT_DATETIME}_logistic_regression_results.txt")
    #with open(results_file, "w") as file_writer:
        #file_writer.write("******* Training Parameters *******\n")
        #file_writer.write(f"Feature Extraction: {VECTOR_TYPE}\n")
        #if VECTOR_TYPE == "tfidf":
        #    file_writer.write(f"TFIDF Parameters: {TFIDF_PARAMS}\n")
        #file_writer.write(f"Training Samples: {len(X_train)}\n")
        #file_writer.write("************************************\n")

        #metrics_train = evaluate_model(model, transformer, X_train, y_train, "Train", file_writer)
        #metrics_test = evaluate_model(model, transformer, X_test, y_test, "Test", file_writer)
        #metrics_dev = evaluate_model(model, transformer, X_dev, y_dev, "Dev", file_writer)

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