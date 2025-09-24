import os
import random
import shutil
from pathlib import Path



# This file creates the TestData folder. Randomly sample from all common names combined


taxons = ["bee", "wasp"]

image_size = "large"

images_folder = f"TrainImages_{image_size}"  
test_folder = f"TestImages_{image_size}"   
test_percentage = 0.15  
os.makedirs(test_folder, exist_ok=True)

for taxon in taxons:
    
    
    train_info_file = f"./TrainDatasetInfo/{taxon}_Info.txt"
    os.makedirs(os.path.dirname(train_info_file), exist_ok=True)

    test_info_file = f"./TestDatasetInfo/{taxon}_Info.txt"
    os.makedirs(os.path.dirname(test_info_file), exist_ok=True)





    species_info = {}
    text_file = f"./AllDatasetInfo/{taxon}_Info.txt"  
    
    with open(text_file, 'r') as file:
        for line in file:
            species, count = line.strip().split(":")
            species_info[species] = int(count)

    all_images = []

    for species, total_count in species_info.items():
        taxon_dir = f"{images_folder}/{taxon}"  
        species_images = list(Path(taxon_dir).glob(f'{species}_*.jpg')) 
        

        if len(species_images) != total_count:
            print(f"Warning: Expected {total_count} images for {species}, but found {len(species_images)}.")

        all_images.extend(species_images)

    if not all_images:
        print(f"Warning: No images found for {taxon}. Skipping this taxon.")
        continue 

    test_count = min(max(1, int(len(all_images) * test_percentage)), len(all_images))  

    test_images = random.sample(all_images, test_count)
    
    taxon_test_dir = os.path.join(test_folder, taxon)
    os.makedirs(taxon_test_dir, exist_ok=True)

    number_of_test_images = 0
    
    for img in test_images:
        try:
            shutil.move(str(img), os.path.join(taxon_test_dir, img.name))

            number_of_test_images +=1
            
        except FileNotFoundError as e:
            print(f"FileNotFoundError: {e}")
    
    with open(test_info_file, "a", encoding='utf-8') as file:
        file.write(f"{species}:{number_of_test_images}\n")

    with open(train_info_file, "a", encoding='utf-8') as file:
        file.write(f"{species}:{total_count - number_of_test_images}\n")        
         

    print(f"Test images successfully moved for species {species} in taxon {taxon}.")
