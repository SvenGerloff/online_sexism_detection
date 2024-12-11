import os
import json
import csv
import pandas as pd
import yaml
import pickle
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, balanced_accuracy_score
from imblearn.over_sampling import SMOTE
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
import wandb
#from utils.load_data import load_processed_data
#TO BE REMOVED
import sys
import os

# Add the absolute path to the utils directory
sys.path.append('/Users/mac/Downloads/online_sexism_detection/utils')

# Now try importing the function
try:
    from load_data import load_processed_data
    print("Import successful!")
except ModuleNotFoundError as e:
    print("Error importing module:", e)

# Define the paths to your parquet files
train_parquet = '/Users/mac/Downloads/online_sexism_detection/data/train.parquet'
dev_parquet = '/Users/mac/Downloads/online_sexism_detection/data/dev.parquet'
test_parquet = '/Users/mac/Downloads/online_sexism_detection/data/test.parquet'

# Use globals() to make these variables accessible in load_data.py
globals()['train_parquet'] = train_parquet
globals()['dev_parquet'] = dev_parquet
globals()['test_parquet'] = test_parquet

# Now call the function
data = load_processed_data()

config_path = os.getenv("CONFIG_PATH", "/Users/mac/Downloads/online_sexism_detection/config.yaml")
with open(config_path, "r") as f:
    config = yaml.safe_load(f)
# Load configuration from YAML
#config_path = os.getenv("CONFIG_PATH", "../config.yaml")
#with open(config_path, "r") as f:
    #config = yaml.safe_load(f)

TFIDF_PARAMS_LIST = config['svm_params']['tfidf_params_list']
WORD2VEC_PARAMS_LIST = config['svm_params']['word2vec_params_list']
MODEL_FOLDER = config['paths']['svm_model_dir']
OUTPUT_FOLDER = config['paths']['output_dir']

def train_and_evaluate_svm_tfidf(X_train, y_train, X_test, y_test):
    wandb.init(project="online_sexism_detection", name="svm_tfidf")

    for tfidf_params in TFIDF_PARAMS_LIST:
        vectorizer = TfidfVectorizer(**tfidf_params)
        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)

        # Apply SMOTE to handle class imbalance
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_train_tfidf, y_train)

        # Train the SVM model
        svm_model = SVC(random_state=42, class_weight='balanced', probability=True)
        svm_model.fit(X_resampled, y_resampled)

        # Make predictions with SVM
        y_pred_svm = svm_model.predict(X_test_tfidf)
        y_pred_proba_svm = svm_model.predict_proba(X_test_tfidf)

        # Create the output DataFrame
        df_output_svm = pd.DataFrame({
            'y_true': y_test,
            'y_pred': y_pred_svm,
            'prob_0': y_pred_proba_svm[:, 0],
            'prob_1': y_pred_proba_svm[:, 1],
            'logit_0': np.nan,
            'logit_1': np.nan
        })

        # Evaluate the SVM model
        print("SVM Results:")
        print(confusion_matrix(y_test, y_pred_svm))
        print(classification_report(y_test, y_pred_svm))

        # Log results
        wandb.log({
            'tfidf_params': tfidf_params,
            "f1": f1_score(y_test, y_pred_svm),
            "balanced_accuracy": balanced_accuracy_score(y_test, y_pred_svm),
            "accuracy": accuracy_score(y_test, y_pred_svm),
        })

        # Save the output DataFrame to a CSV file 
        df_output_svm.to_csv(os.path.join(OUTPUT_FOLDER, f'svm_predictions_{tfidf_params["max_features"]}.csv'), index=False)

    wandb.finish()

def train_and_evaluate_svm_word2vec(X_train, y_train, X_test, y_test):
    wandb.init(project="online_sexism_detection", name="word2vec_svm")

    for word2vec_params in WORD2VEC_PARAMS_LIST:
        X_processed_train = X_train.apply(lambda x: x.split()).tolist()
        X_processed_test = X_test.apply(lambda x: x.split()).tolist()

        word2vec_model = Word2Vec(sentences=X_processed_train, **word2vec_params)

        def vectorize_sentences(sentences, model):
            vectors = []
            for sentence in sentences:
                word_vectors = [model.wv[word] for word in sentence if word in model.wv]
                if word_vectors:
                    vectors.append(np.mean(word_vectors, axis=0))
                else:
                    vectors.append(np.zeros(model.vector_size))
            return np.array(vectors)

        X_train_vectors = vectorize_sentences(X_processed_train, word2vec_model)
        X_test_vectors = vectorize_sentences(X_processed_test, word2vec_model)

        smote = SMOTE(random_state=42)
        X_resampled_w2v, y_resampled_w2v = smote.fit_resample(X_train_vectors, y_train)

        # Train the SVM model with Word2Vec
        svm_model_w2v = SVC(random_state=42, class_weight='balanced', probability=True)
        svm_model_w2v.fit(X_resampled_w2v, y_resampled_w2v)

        # Make predictions with SVM (Word2Vec)
        y_pred_svm_w2v = svm_model_w2v.predict(X_test_vectors)

        # Evaluate
        print(f"Word2Vec Params: {word2vec_params}")
        print("Accuracy:", accuracy_score(y_test, y_pred_svm_w2v))
        print(classification_report(y_test, y_pred_svm_w2v))

        # Log results to Weights & Biases
        wandb.log({
            'word2vec_params': word2vec_params,
            "f1": f1_score(y_test, y_pred_svm_w2v),
            "balanced_accuracy": balanced_accuracy_score(y_test, y_pred_svm_w2v),
            "accuracy": accuracy_score(y_test, y_pred_svm_w2v),
        })

        # Save the output DataFrame to a CSV file
        df_output_svm_w2v = pd.DataFrame({
            'y_true': y_test,
            'y_pred': y_pred_svm_w2v,
            'logit_0': np.nan,
            'logit_1': np.nan
        })
        df_output_svm_w2v.to_csv(os.path.join(OUTPUT_FOLDER, f'svm_w2v_predictions_{word2vec_params["vector_size"]}.csv'), index=False)

    wandb.finish()

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