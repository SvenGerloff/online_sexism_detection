import os
import pandas as pd
import stanza
import nltk
import re
from tqdm import tqdm
import yaml
from stanza.utils.conll import CoNLL
import requests

# Load configuration from YAML
with open("../config.yaml", "r") as f:
    config = yaml.safe_load(f)

# URL from configuration
data_url = config["urls"]["data_url"]
# Paths relative to the current working directory
base_dir = os.getcwd()
# Input data path and output directory
data_path = os.path.normpath(os.path.join(base_dir, config["paths"]["data_file"]))
output_folder = os.path.normpath(os.path.join(base_dir, config["paths"]["output_dir"]))
# Main output files
parquet_path = os.path.join(output_folder, config["files"]["parquet_file"])
conllu_path = os.path.join(output_folder, config["files"]["conllu_file"])
# Subset files for train, dev, and test
train_parquet = os.path.join(output_folder, config["files"]["subsets"]["train"]["parquet"])
train_conllu = os.path.join(output_folder, config["files"]["subsets"]["train"]["conllu"])
dev_parquet = os.path.join(output_folder, config["files"]["subsets"]["dev"]["parquet"])
dev_conllu = os.path.join(output_folder, config["files"]["subsets"]["dev"]["conllu"])
test_parquet = os.path.join(output_folder, config["files"]["subsets"]["test"]["parquet"])
test_conllu = os.path.join(output_folder, config["files"]["subsets"]["test"]["conllu"])

# Initialize tqdm for pandas
tqdm.pandas()

# Initialize Stanza and NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
stanza.download('en')
nlp_pipeline = stanza.Pipeline(lang='en')


def load_data_and_prepare():
    # Ensure the directory for the data path exists
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    # Attempt to download the data from the URL first
    try:
        query_parameters = {"downloadformat": "csv"}
        download = requests.get(data_url, params=query_parameters)
        download.raise_for_status()

        # Save downloaded content to the specified local path
        with open(data_path, mode="wb") as file:
            file.write(download.content)

        print(f"Data downloaded and saved to {data_path}.")

        # Load CSV data directly from the downloaded file
        df = pd.read_csv(data_path)

    except requests.RequestException:
        print(f"Failed to download data from {data_url}.")

        # Check if the local file exists
        if os.path.exists(data_path):
            print(f"Loading data from local file at {data_path}...")
            df = pd.read_csv(data_path)
        else:
            print(f"Error: Local file not found at {data_path}.")
            return None  # Return None if both download and local loading fail

    # Prepare the data
    df['label_sexist'] = df['label_sexist'].map({'sexist': 1, 'not sexist': 0})
    df = df.loc[:, ["text", "label_sexist", "split"]]
    df.rename(columns={'label_sexist': 'label'}, inplace=True)
    return df

def clean_text(text):
    return re.sub(r'\[USER\]', '', text).strip()

def process_pipeline(text):
    """Clean and process a single text entry to extract lemmas and POS tags."""
    cleaned_text = clean_text(text)
    doc = nlp_pipeline(cleaned_text)
    lemmas, pos_tags = [], []

    for sentence in doc.sentences:
        for word in sentence.words:
            lemmas.append(word.lemma)
            pos_tags.append(word.upos)

    return doc, lemmas, pos_tags

def process_text(df):
    """Process a DataFrame of text entries to extract lemmas and POS tags with progress bar."""
    # Apply process_pipeline to each text entry in the DataFrame with progress tracking
    processed = df['text'].progress_apply(process_pipeline)

    # Unpack the results into separate lists
    docs, lemmas, pos_tags = zip(*processed)

    # Add the lists to the DataFrame as new columns
    df = df.copy()  # Ensure we're working with a copy
    df['lemma'] = lemmas
    df['pos'] = pos_tags

    return docs, df

def save_to_conllu(docs, output_path_conllu):
    with open(output_path_conllu, 'w', encoding='utf-8') as f:
        for doc in docs:
            CoNLL.write_doc2conll(doc, f)

def save_to_parquet(df, output_path_df):
    """Save df as a Parquet file"""
    df.to_parquet(output_path_df, index=False)

def prepare_full_dataset():
    """Load, process, and save the entire dataset"""
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Load and process data
    df = load_data_and_prepare()
    docs, df = process_text(df)

    save_to_parquet(df, parquet_path)
    save_to_conllu(docs, conllu_path)

    return df, docs

def prepare_split_datasets():
    """Split dataset by train, dev, and test subsets"""
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    df = load_data_and_prepare()
    split_datasets = {}

    # Dictionary to store paths for each split type
    paths = {
        "train": {"parquet": train_parquet, "conllu": train_conllu},
        "dev": {"parquet": dev_parquet, "conllu": dev_conllu},
        "test": {"parquet": test_parquet, "conllu": test_conllu}
    }
    for split_type in ['train', 'dev', 'test']:
        subset_df = df[df['split'] == split_type]
        if not subset_df.empty:
            docs, processed_subset = process_text(subset_df)

            # Save each subset as Parquet and CoNLL-U
            save_to_parquet(processed_subset, paths[split_type]["parquet"])
            save_to_conllu(docs, paths[split_type]["conllu"])

            # Store processed subset in dictionary
            split_datasets[split_type] = (processed_subset, docs)

    return split_datasets

def load_processed_data(split=None):
    # Load the full dataset if no specific split is requested
    if split is None:
        if os.path.exists(parquet_path):
            return pd.read_parquet(parquet_path)
        else:
            print("Warning: Full dataset file not found.")
            return None

    # Load specified split datasets
    split_dataframes = {}
    paths = {
        "train": train_parquet,
        "dev": dev_parquet,
        "test": test_parquet,
    }
    # Load each specified split from paths dictionary
    for split_type in split:
        split_dataset_path = paths.get(split_type)
        if split_dataset_path and os.path.exists(split_dataset_path):
            split_dataframes[split_type] = pd.read_parquet(split_dataset_path)
        else:
            print(f"Warning: {split_type} split file not found.")

    return split_dataframes