import os
import pandas as pd
import yaml

# Determine the path to config.yaml
config_path = os.getenv("CONFIG_PATH", "../config.yaml")

# Load configuration from YAML
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

# Paths relative to the current working directory
base_dir = os.getcwd()
# Input data_submission path and output directory
output_folder = os.path.normpath(os.path.join(base_dir, config["paths"]["output_dir"]))
# Subset files for train, dev, and test
train_parquet = os.path.join(output_folder, config["files"]["subsets"]["train"]["parquet"])
dev_parquet = os.path.join(output_folder, config["files"]["subsets"]["dev"]["parquet"])
test_parquet = os.path.join(output_folder, config["files"]["subsets"]["test"]["parquet"])

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