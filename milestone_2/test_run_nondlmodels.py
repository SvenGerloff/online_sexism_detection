import os
import re
import ast
import json
import csv
import yaml
import pickle
import pandas as pd
from datetime import datetime
from sklearn.metrics import f1_score, balanced_accuracy_score, accuracy_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from utils.load_data import load_processed_data

# Remove non-standard unicode surrogates
surrogate_pattern = re.compile(r'[\ud800-\udfff]')
def remove_surrogates(text):
    return surrogate_pattern.sub('', text) if isinstance(text, str) else text

# Convert string representations of token lists into actual lists
def to_token_list(x):
    if isinstance(x, list):
        return x
    elif isinstance(x, str):
        x = x.strip()
        if x.startswith('[') and x.endswith(']'):
            try:
                parsed = ast.literal_eval(x)
                if isinstance(parsed, list):
                    return parsed
            except:
                pass
        return x.split()
    return str(x).split()

class BaseClassifier:
    def __init__(self, classifier_type, config_path="../config.yaml"):
        self.classifier_type = classifier_type
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        # Load parameters for this classifier
        param_key = f"{self.classifier_type}_params"
        if param_key not in self.config:
            raise ValueError(f"No parameters for {self.classifier_type} in config.")

        self.params = self.config[param_key]
        self.TRAIN_MODEL = self.params.get("train", False)
        self.MODEL_TIMESTAMP = self.params.get("model_timestamp")
        self.VECTOR_TYPE = self.params.get("feature_extraction", "tfidf")
        self.TFIDF_PARAMS = self.params.get("tfidf_params", {})

        # Set up directories and file names
        base_folder = self.config['paths']['model_dir']
        self.CURRENT_DATETIME = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.MODEL_FOLDER = os.path.join(base_folder, self.classifier_type)
        os.makedirs(self.MODEL_FOLDER, exist_ok=True)
        self.model_name = f"{self.classifier_type}_model"

        # Initialize vectorizer if using TF-IDF
        self.transformer = None
        if self.VECTOR_TYPE == "tfidf":
            self.vectorizer = TfidfVectorizer(
                max_features=self.TFIDF_PARAMS.get('max_features'),
                ngram_range=tuple(self.TFIDF_PARAMS.get('ngram_range', (1, 1))),
                min_df=self.TFIDF_PARAMS.get('min_df', 1)
            )

    def load_data(self):
        data = load_processed_data()
        self.X_train, self.y_train = data["train"]["lemma"], data["train"]["label"]
        self.X_test, self.y_test = data["test"]["lemma"], data["test"]["label"]
        self.X_dev, self.y_dev = data["dev"]["lemma"], data["dev"]["label"]

    def create_model(self):
        raise NotImplementedError

    def train(self):
        if self.VECTOR_TYPE == "tfidf":
            X_train_vec = self.vectorizer.fit_transform(self.X_train)
            self.transformer = self.vectorizer
        else:
            X_train_vec = self.X_train

        self.model = self.create_model()
        self.model.fit(X_train_vec, self.y_train)

        model_file = os.path.join(self.MODEL_FOLDER, f"{self.model_name}_{self.CURRENT_DATETIME}.pkl")
        with open(model_file, "wb") as f:
            pickle.dump((self.model, self.transformer), f)
        print(f"Model trained and saved: {model_file}")

    def load_model(self):
        if not self.MODEL_TIMESTAMP:
            raise ValueError("MODEL_TIMESTAMP not set in config.")
        model_path = os.path.join(self.MODEL_FOLDER, f"{self.model_name}_{self.MODEL_TIMESTAMP}.pkl")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        with open(model_path, "rb") as f:
            print(f"Loading model: {model_path}")
            self.model, self.transformer = pickle.load(f)

    def evaluate(self, X_data, y_data, dataset_name, file_writer=None):
        X_vec = self.transformer.transform(X_data) if self.transformer else X_data
        y_pred = self.model.predict(X_vec)
        metrics = {
            "accuracy": accuracy_score(y_data, y_pred),
            "balanced_accuracy": balanced_accuracy_score(y_data, y_pred),
            "f1": f1_score(y_data, y_pred),
            "classification_report": classification_report(y_data, y_pred),
            "confusion_matrix": confusion_matrix(y_data, y_pred).tolist()
        }

        # Optionally write to results file
        if file_writer:
            file_writer.write(f"\n******* {dataset_name} Metrics *******\n")
            file_writer.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
            file_writer.write(f"Balanced Accuracy: {metrics['balanced_accuracy']:.4f}\n")
            file_writer.write(f"F1 Score: {metrics['f1']:.4f}\n")
            file_writer.write("\nClassification Report:\n")
            file_writer.write(metrics["classification_report"])
            file_writer.write("\nConfusion Matrix:\n")
            for row in metrics["confusion_matrix"]:
                file_writer.write(f"{row}\n")
            file_writer.write("************************************\n")

        # Print to console
        print(f"\n{dataset_name} Metrics:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
        print(f"F1 Score: {metrics['f1']:.4f}")
        print("Classification Report:\n", metrics["classification_report"])
        print("Confusion Matrix:", metrics["confusion_matrix"])

        return metrics

    def save_dataset_output(self, X_data, y_data, dataset_name):
        lem_tokens_lists = X_data.apply(to_token_list)
        X_vec = self.transformer.transform(X_data) if self.transformer else X_data
        y_pred = self.model.predict(X_vec)

        # Extract tokens used by the model
        if self.transformer:
            feature_names = self.transformer.get_feature_names_out()
            model_tokens_list = []
            for i in range(X_vec.shape[0]):
                token_indices = X_vec[i].nonzero()[1]
                model_tokens_list.append(json.dumps([feature_names[idx] for idx in token_indices]))
        else:
            model_tokens_list = [json.dumps(tokens) for tokens in lem_tokens_lists]

        df_output = pd.DataFrame({
            "lem_tokens": lem_tokens_lists.apply(lambda t: json.dumps(t)),
            "model_tokens": model_tokens_list,
            "y_pred": y_pred,
            "y_true": y_data
        }).applymap(remove_surrogates)

        output_file = f"{self.classifier_type}_{dataset_name}_output_{self.CURRENT_DATETIME}.csv"
        df_output.to_csv(
            os.path.join(self.MODEL_FOLDER, output_file),
            index=False, encoding='utf-8', sep=',',
            quoting=csv.QUOTE_MINIMAL, quotechar="'", escapechar='\\'
        )
        print(f"{dataset_name} output saved: {output_file}")

    def run(self):
        self.load_data()
        results_file = os.path.join(self.MODEL_FOLDER, f"{self.model_name}_results_{self.CURRENT_DATETIME}.txt")

        # Write parameters to results file
        with open(results_file, "w") as f:
            f.write("******* Training Parameters *******\n")
            f.write(f"Classifier Type: {self.classifier_type}\n")
            for k, v in self.params.items():
                f.write(f"{k}: {v}\n")
            f.write("************************************\n")

        # Train or load model
        if self.TRAIN_MODEL:
            print("Training the model...")
            self.train()
        else:
            print("Loading existing model...")
            self.load_model()

        # Evaluate on all sets
        with open(results_file, "a") as f:
            if not self.TRAIN_MODEL:
                f.write("\nEvaluating existing model:\n")
            self.evaluate(self.X_train, self.y_train, "Train", f)
            self.evaluate(self.X_test, self.y_test, "Test", f)
            self.evaluate(self.X_dev, self.y_dev, "Dev", f)

        # Save model predictions
        self.save_dataset_output(self.X_train, self.y_train, "Train")
        self.save_dataset_output(self.X_test, self.y_test, "Test")
        self.save_dataset_output(self.X_dev, self.y_dev, "Dev")

        print(f"Results saved to {results_file}")

class NaiveBayesClassifier(BaseClassifier):
    def __init__(self, config_path="../config.yaml"):
        super().__init__("naivebayes", config_path)

    def create_model(self):
        return MultinomialNB()

class XGBoostClassifier(BaseClassifier):
    def __init__(self, config_path="../config.yaml"):
        super().__init__("xgboost", config_path)

    def create_model(self):
        return XGBClassifier(
            max_depth=self.params.get("max_depth", 6),
            learning_rate=self.params.get("learning_rate", 0.1),
            n_estimators=self.params.get("n_estimators", 100),
            eval_metric=self.params.get("eval_metric", "logloss"),
            use_label_encoder=False
        )

class RandomForestClassifierWrapper(BaseClassifier):
    def __init__(self, config_path="../config.yaml"):
        super().__init__("randomforest", config_path)

    def create_model(self):
        return RandomForestClassifier(
            n_estimators=self.params.get("n_estimators", 100),
            random_state=self.params.get("random_state", 42)
        )

def main():
    NaiveBayesClassifier().run()
    XGBoostClassifier().run()
    RandomForestClassifierWrapper().run()

if __name__ == "__main__":
    main()