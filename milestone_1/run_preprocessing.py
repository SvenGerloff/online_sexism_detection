import sys
import pandas as pd

# Add the parent directory to the system path
sys.path.append('..')
# Import functions from preprocessing module
from utils.preprocessing import (
    prepare_split_datasets,
    load_processed_data,
    load_conllu_data,
    read_conllu_file,
    load_conllu_datasets
)

# Uncomment the lines you want to execute - the next lines will overwrite the data:

# Prepare the full dataset
# prepare_split_datasets()



# Prepare a sample dataset with 10 samples
# This function processes and splits the data, taking a sample of n_samples
#prepare_split_datasets(n_samples=10)

# Load the processed data for the 'train', 'dev', and 'test' splits
# split_datasets = load_processed_data(split=['train', 'dev', 'test'])
#
# # Display information about the training dataset
# print("Training Dataset Info:")
# split_datasets['train'].info()
# print("\nFirst 10 rows of the Training Dataset:")
# print(split_datasets['train'].head(10))
#
# # Display information about the development dataset
# print("\nDevelopment Dataset Info:")
# split_datasets['dev'].info()
# print("\nFirst 10 rows of the Development Dataset:")
# print(split_datasets['dev'].head(10))
#
# # Display information about the test dataset
# print("\nTest Dataset Info:")
# split_datasets['test'].info()
# print("\nFirst 10 rows of the Test Dataset:")
# print(split_datasets['test'].head(10))
#
# # Load the CONLLU formatted data for the splits
# split_docs = load_conllu_data(split=['train', 'dev', 'test'])
#
# # Print the CONLLU data for each split
# print("\nTraining CONLLU Data:")
# print(split_docs['train'])
#
# print("\nDevelopment CONLLU Data:")
# print(split_docs['dev'])
#
# print("\nTest CONLLU Data:")
# print(split_docs['test'])