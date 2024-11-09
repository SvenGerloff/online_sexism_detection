import os
import pandas as pd
import stanza
import nltk
import re
from tqdm import tqdm
import yaml
from stanza.utils.conll import CoNLL
import requests
from nltk.corpus import stopwords

# Determine the path to config.yaml
config_path = os.getenv("CONFIG_PATH", "../config.yaml")

# Load configuration from YAML
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.realpath(__file__))

# URL from configuration
data_url = config["urls"]["data_url"]
# Paths relative to the current working directory
base_dir = os.getcwd()
# Input data_submission path and output directory
data_path = os.path.normpath(os.path.join(base_dir, config["paths"]["data_file"]))
data_path_csv = os.path.normpath(os.path.join(base_dir, config["paths"]["data_file_csv"]))
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
nlp_pipeline = stanza.Pipeline('en', processors='tokenize,lemma,pos')

def load_data_and_prepare():
    # Ensure the directory for the data path exists
    os.makedirs(os.path.dirname(data_path), exist_ok=True)

    # Check if the .parquet file already exists
    if os.path.exists(data_path):
        print(f"Loading data from {data_path}...")
        df = pd.read_parquet(data_path)
    else:
        # Attempt to download the data
        try:
            query_parameters = {"downloadformat": "csv"}
            download = requests.get(data_url, params=query_parameters)
            download.raise_for_status()

            # Save downloaded content as .csv file
            with open(data_path_csv, mode="wb") as file:
                file.write(download.content)

            # Read the CSV data into a DataFrame
            df = pd.read_csv(data_path_csv)

            # Save DataFrame as .parquet and remove the .csv file
            df.to_parquet(data_path, index=False)
            os.remove(data_path_csv)
            print(f"Data downloaded and saved to {data_path}.")

        except requests.RequestException as e:
            print(f"Failed to download data from {data_url}: {e}")
            return None

    # Prepare the data
    df['label_sexist'] = df['label_sexist'].map({'sexist': 1, 'not sexist': 0})
    df = df[["text", "label_sexist", "split"]]
    df.rename(columns={'label_sexist': 'label'}, inplace=True)
    return df

def clean_text(text):
    """Clean the text by removing [USER] and [URL] tags, and count their occurrences."""
    user_count = len(re.findall(r'\[USER\]', text))
    url_count = len(re.findall(r'\[URL\]', text))
    cleaned_text = re.sub(r'\[USER\]', 'USERTOKEN', text).strip()
    cleaned_text = re.sub(r'\[URL\]', 'URLTOKEN', cleaned_text).strip()
    return cleaned_text, user_count, url_count

def process_pipeline(text):
    """Clean and process a single text entry to extract lemmas and POS tags."""

    cleaned_text, user_count, url_count = clean_text(text)
    doc = nlp_pipeline(cleaned_text)
    lemmas, pos_tags = [], []

    custom_stopwords = ['what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those',
                        'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
                        'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through',
                        'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on',
                        'off', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where',
                        'why', 'how', 'such', 'so', 'than', 'very', 'will', 'just', 'now', 'd', 'll', 'm', 'o', 're',
                        've', 'y', 'ain']

    for sentence in doc.sentences:
        for word in sentence.words:
            # filter stopwords
            if word.lemma in custom_stopwords:
                continue
            # filter punctuation
            if not re.search(r'[a-zA-Z]', word.lemma):
                continue

            lemmas.append(word.lemma)
            pos_tags.append(word.upos)

    return doc, lemmas, pos_tags, user_count, url_count

def process_text(df):
    """Process a DataFrame of text entries to extract lemmas, POS tags, and counts of [USER] and [URL]."""
    # Apply process_pipeline to each text entry in the DataFrame with progress tracking
    processed = df['text'].progress_apply(process_pipeline)

    # Unpack the results into separate lists
    docs, lemmas, pos_tags, user_counts, url_counts = zip(*processed)

    # Add the lists to the DataFrame as new columns
    df = df.copy()
    df['lemma'] = lemmas
    df['pos'] = pos_tags
    df['user_count'] = user_counts
    df['url_count'] = url_counts

    return docs, df

def save_to_conllu(docs, output_path_conllu):
    with open(output_path_conllu, 'w', encoding='utf-8') as f:
        for doc in docs:
            CoNLL.write_doc2conll(doc, f)

def save_to_parquet(df, output_path_df):
    """Save df as a Parquet file"""
    df.to_parquet(output_path_df, index=False)


def prepare_split_datasets(n_samples=None):
    """ Split dataset by train, dev, and test subsets and save each subset as Parquet and CoNLL-U files."""
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Load and prepare the full dataset
    df = load_data_and_prepare()
    split_datasets = {}

    # Paths for each split type
    paths = {
        "train": {"parquet": train_parquet, "conllu": train_conllu},
        "dev": {"parquet": dev_parquet, "conllu": dev_conllu},
        "test": {"parquet": test_parquet, "conllu": test_conllu}
    }

    for split_type in ['train', 'dev', 'test']:
        subset_df = df[df['split'] == split_type]

        # If n_samples is specified, take a sample of that size
        if n_samples is not None:
            subset_df = subset_df.sample(n=min(n_samples, len(subset_df)), random_state=42)

        if not subset_df.empty:
            # Process each subset
            docs, processed_subset = process_text(subset_df)

            # Save each subset as Parquet and CoNLL-U files
            save_to_parquet(processed_subset, paths[split_type]["parquet"])
            print('')
            print(f'DataFrame {split_type} was saved as a Parquet file at: {paths[split_type]["parquet"]}')
            save_to_conllu(docs, paths[split_type]["conllu"])
            print(f'Document {split_type} was saved as a CoNLL-U file at: {paths[split_type]["conllu"]}')

            # Store processed subset in dictionary
            split_datasets[split_type] = (processed_subset, docs)

    return split_datasets

def load_processed_data(split=None):
    # Load the full dataset if no specific split is requested
    if split is None:
        split = ["train", "dev", "test"]

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
            print(f"df: {split_type.capitalize()} split loaded.")
        else:
            print(f"Warning: {split_type} split file not found.")

    return split_dataframes

def load_conllu_data(split=None):
    """Load CoNLL files, either in full or by specified splits."""
    # Define paths for each split
    if split is None:
        split = ["train", "dev", "test"]

    paths = {
        "train": train_conllu,
        "dev": dev_conllu,
        "test": test_conllu,
    }

    # Load each specified split
    split_docs = {}
    for split_type in split:
        split_path = paths.get(split_type)
        if split_path and os.path.exists(split_path):
            docs = CoNLL.conll2doc(split_path)
            print(f"CoNLL file: {split_type.capitalize()} split loaded.")
            split_docs[split_type] = docs
        else:
            print(f"Warning: {split_type} split file not found at {split_path}.")

    return split_docs


def load_conllu_datasets(split=None):
    """ Load CoNLL-U files for 'train', 'dev', and 'test' splits and return DataFrames for each."""
    if split is None:
        split = ["train", "dev", "test"]

    # Define paths for each split
    paths = {
        "train": train_conllu,
        "dev": dev_conllu,
        "test": test_conllu,
    }

    dataframes = {}

    for split, path in paths.items():
        if os.path.exists(path):
            print(f"Loading {split} data from {path}...")
            df = read_conllu_file(path)
            dataframes[split] = df
            print(f"{split.capitalize()} CoNLL-U files as DataFrame loaded successfully with {len(df)} rows.")
        else:
            print(f"Warning: {split} file not found at {path}.")

    return dataframes.get("train"), dataframes.get("dev"), dataframes.get("test")


def read_conllu_file(file_path):

    sentences = []
    current_sentence = []
    
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if line.startswith('#'):
                continue
            if line == '':
                if current_sentence:
                    sentences.append(current_sentence)
                    current_sentence = []
                continue
            
            parts = line.split('\t')
            if len(parts) != 10:
                continue 

            word_id = parts[0]
            form = parts[1]
            lemma = parts[2]
            upos = parts[3]
            xpos = parts[4]
            feats = parts[5]
            head = parts[6]
            deprel = parts[7]
            start_char = parts[8]
            end_char = parts[9]

            current_sentence.append({
                'id': word_id,
                'form': form,
                'lemma': lemma,
                'upos': upos,
                'xpos': xpos,
                'feats': feats,
                'head': head,
                'deprel': deprel,
                'start_char': start_char,
                'end_char': end_char,
            })

        if current_sentence:  # Add the last sentence if it exists
            sentences.append(current_sentence)

    df = pd.DataFrame([{
        'id': word['id'],
        'form': word['form'],
        'lemma': word['lemma'],
        'upos': word['upos'],
        'xpos': word['xpos'],
        'feats': word['feats'],
        'head': word['head'],
        'deprel': word['deprel'],
        'start_char': word['start_char'],
        'end_char': word['end_char'],
    } for sentence in sentences for word in sentence])

    return df
