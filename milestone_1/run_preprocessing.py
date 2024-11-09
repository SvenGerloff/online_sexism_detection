import sys
import pandas as pd

sys.path.append('..')
from utils.preprocessing import (
    prepare_full_dataset,
    prepare_split_datasets,
    load_processed_data,
    read_conllu_file  # Import the new function
)


processed_df, conllu_format = prepare_full_dataset()
print("Processed DataFrame:")
print(processed_df)


split_datasets = prepare_split_datasets()


split_dfs = load_processed_data(split=['train', 'dev', 'test'])
print("\nTrain DataFrame:")
print(split_dfs["train"])
print("\nDev DataFrame:")
print(split_dfs["dev"])
print("\nTest DataFrame:")
print(split_dfs["test"])
