import pandas as pd
import os
from glob import glob

# Set your folder path here
folder_path = "C:\\Users\\6\\PycharmProjects\\CESGCN_Code_Pipeline_modified\\outbreak_files"

# Find all CSV files in the folder
csv_files = glob(os.path.join(folder_path, "*.csv"))

# Read and concatenate all CSV files
merged_df = pd.concat([pd.read_csv(file) for file in csv_files], ignore_index=True)

# Save to a single output file
merged_df.to_csv("merged_output.csv", index=False)
