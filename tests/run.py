import os
import subprocess

# Define the path to the "ops" folder
ops_folder = "tests/Ops/"
assigner_binary = "build/src/mlir-assigner"

# Get a list of all files and folders within the "ops" folder
items = os.listdir(ops_folder)

# Iterate over the list and check if each item is a folder
subfolders = []
for item in items:
    item_path = os.path.join(ops_folder, item)
    if os.path.isdir(item_path):
        subfolders.append(item)

# Iterate over each subfolder
for subfolder in subfolders:
    subfolder_path = os.path.join(ops_folder, subfolder)
    # Get a list of all files within the subfolder
    files = os.listdir(subfolder_path)
    # Iterate over the files and grab those ending in ".mlir"
    for file in files:
        if file.endswith(".mlir"):
            # Construct the JSON file name by replacing the ".mlir" extension with ".json"
            json_file = file.replace(".mlir", ".json")
            json_file_path = os.path.join(subfolder_path, json_file)
            # Call the assigner binary with the input files
            subprocess.run([assigner_binary, file, json_file_path])
            

