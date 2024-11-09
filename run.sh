#!/bin/bash
# Check if conda is installed
if ! command -v conda &> /dev/null
then
    echo "Conda not found"
    exit 1
fi

# Create the conda environment based on the environment file
echo "Creating the conda environment from sexism_detection.yml..."
conda env create -f sexism_detection.yml

# Set the absolute path of config.yaml as an environment variable
export CONFIG_PATH="$(pwd)/config.yaml"

# Extract the preprocessing file path from config.yaml
PREPROCESSING_FILE=$(python -c "
import yaml
with open('$CONFIG_PATH', 'r') as f:
    config = yaml.safe_load(f)
print(config['paths']['preprocessing_file'])
")

# Add utils directory to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)/utils"

# Run the preprocessing file within the conda environment with unbuffered output
echo "Running the preprocessing script..."
conda run -n sexism_detection python -u "$PREPROCESSING_FILE"