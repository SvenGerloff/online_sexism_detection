# Online Sexism Detection - Github repository

## Group 21: Clarity Crew

Members: Sven Gerloff, Theresa Mayer, Yasmine Khajjou, Thomas Klar

## Setup

- Create new python environment
- conda install -r requirements.txt
- conda install -c standfordnlp stanza

## File Structure

- All data files go directly into \data
- Put helper functions into .py files in \utils and import them into notebooks

## Milestone 1

### Guide to Preprocess the Data

1. Complete the steps in the Setup section to prepare the environment and install all required packages.
2. Navigate to `milestone_1/run_preprocessing`.
3. Uncomment the lines corresponding to the preprocessing steps you’d like to run.
4. Use the `prepare_split_datasets()` method to preprocess the entire dataset. This function:
   - Downloads the data from the [Git repository](https://github.com/rewire-online/edos) if not already available.
   - Prepares the data and splits it into Train, Dev, and Test sets.
   - Saves each split as a `.parquet` file and a `.conllu` file in the `data_submission` folder.

**Note**: Processing the full dataset takes approximately 2 hours. To test the function quickly, use the `n_samples` parameter to specify the number of samples per split to process. Other methods, such as `load_processed_data` and `load_conllu_data`, allow loading the processed files for any specified splits or for all splits.
