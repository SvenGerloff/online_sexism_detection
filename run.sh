#!/bin/bash
# Check if conda is installed
if ! command -v conda &> /dev/null
then
    echo "Conda not found. Please install Anaconda or Miniconda and try again."
    exit 1
fi

# Create the conda environment based on the environment file
echo "Creating the conda environment from sexism_detection.yml..."
conda env create -f sexism_detection.yml

# Activate the environment
echo "Activating the environment 'sexism_detection'..."
conda activate sexism_detection || source activate sexism_detection

# Extract the preprocessing file path from config.yaml
PREPROCESSING_FILE=$(python -c "
import yaml
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)
print(config['paths']['preprocessing_file'])
")

# Run the preprocessing file
echo "Running the preprocessing script..."
python "$PREPROCESSING_FILE"