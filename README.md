# Online Sexism Detection

## Group 21: Clarity Crew

Members: Sven Gerloff, Theresa Mayer, Yasmine Khajjou, Thomas Klar

## Setup

### Option 1:

To set up the `sexism_detection` environment using the provided `sexism_detection.yml` file.

1. Ensure Conda is Installed

2. Create the Environment: Run the following command in your terminal to create the Conda environment using the `sexism_detection.yml` file:

   ```bash
   conda env create -f sexism_detection.yml

3. Activate the Environment

   ```bash
   conda activate sexism_detection

### Option 2:

- Create new python environment
- conda install -r requirements.txt
- conda install -c standfordnlp stanza

## File Structure

- All data files go directly into `data_submission/`
- Put helper functions into `.py` files in `utils/` and import them into notebooks.
- The `preprocessing.py` file in `utils/` contains functions for loading, cleaning, and processing the dataset, including splitting the data into train, dev, and test sets and saving the processed output in `.parquet` and `.conllu` formats.

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

### Data Exploration

In the file `Milestone_1/01_data_exploration.ipynb`, you can find the various data exploration steps we applied. This notebook also includes examples on how to read the stored data and use it.

### Report - Milestone 1 (PDF)

The file `Milestone_1_Report.pdf` provides a detailed description of the preprocessing steps we applied, including the rationale behind each step. It also outlines the main findings from our data exploration.

## Milestone 2

### Experiment Setup

### Models

#### Deep-Learning Models

#### Non-Deep-Learning Models

### Report - Milestone 2 (PDF)

The file `Milestone_2_Report.pdf` provides a detailed description of the steps we applied, including the rationale behind each one. It also outlines the main findings from our baseline models.


