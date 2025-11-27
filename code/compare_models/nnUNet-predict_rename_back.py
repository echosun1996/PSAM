from ast import arg
import csv
import os
from rich import print
from tqdm import tqdm

# /home/jiajun/zu52/PSAM/data/nnUNetFrame/Atlas_test/image_file_mapping.csv


import pandas as pd
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--processing-path", type=str, help="Path to the processing")
parser.add_argument("--csv-path", type=str, help="csv output path")
args = parser.parse_args()


csv_file_path = args.csv_path
process_path = args.processing_path

# Read the CSV file
df = pd.read_csv(csv_file_path)

# Create a dictionary mapping Covert File Name to Original Image Name
name_mapping = dict(zip(df["Covert File Name"], df["Original Image Name"]))

# List the files in the process_path
for file_name in tqdm(os.listdir(process_path)):
    file_path = os.path.join(process_path, file_name)
    if not file_name.endswith(".png"):
        continue
    if file_name in name_mapping:
        # Get the original name
        original_name = name_mapping[file_name]
        original_path = os.path.join(process_path, original_name)
        # Rename the file
        os.rename(file_path, original_path)
    else:
        print(f"File {file_name} not found in the CSV file: {csv_file_path}")
        exit(-1)
