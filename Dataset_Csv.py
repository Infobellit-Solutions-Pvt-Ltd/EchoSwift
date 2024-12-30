import pandas as pd
import glob
from pathlib import Path

# Path to the Input_Dataset directory
dataset_dir = Path("Input_Dataset")
# Output file
output_file = dataset_dir / "Dataset.csv"

# List to hold DataFrames
dataframes = []

# Find all Dataset_*.csv files
dataset_files = glob.glob(str(dataset_dir / "Dataset_*.csv"))

# Read each file and append to the list
for file in dataset_files:
    df = pd.read_csv(file)
    dataframes.append(df)

# Concatenate all DataFrames into one
consolidated_df = pd.concat(dataframes, ignore_index=True)

# Save to Dataset.csv
consolidated_df.to_csv(output_file, index=False)

print(f"Consolidated dataset created at: {output_file}")