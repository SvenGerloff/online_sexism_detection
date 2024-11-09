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
#prepare_split_datasets()
