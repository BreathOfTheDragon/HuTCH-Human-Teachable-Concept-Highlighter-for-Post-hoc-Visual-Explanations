import requests
import os
import json
import time
import os.path




taxons = ["bee", "wasp"]


def SortDatasetInfo(taxon):
    file_path = f'./AllDatasetInfo/{taxon}_Info.txt'
    all_species = set()
    
        
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            for line in f:
                all_species.add(line.strip())
                print(f"Line: {line}")

    all_species_sorted = sorted(all_species)

    with open(file_path, 'w') as file:
        for member in all_species_sorted:
            file.writelines(member + "\n")

    print(f"The contents of {file_path} have been updated.")


    
for taxon in taxons:
    SortDatasetInfo(taxon)