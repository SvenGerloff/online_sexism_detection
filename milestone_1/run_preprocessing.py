import sys

sys.path.append('..')
from utils.preprocessing import prepare_full_dataset, prepare_split_datasets,load_processed_data


processed_df, conllu_format = prepare_full_dataset()
split_datasets = prepare_split_datasets()

split_dfs = load_processed_data(split=['train', 'dev', 'test'])
print(split_dfs["train"])
print(split_dfs["dev"])
print(split_dfs["test"])


