import requests
import os
import json
import time
from pathlib import Path


taxon = "wasp"


info_file = f"../AllDatasetInfo/{taxon}_Info.txt"
if os.path.exists(info_file):
    os.remove(info_file)

common_name = "{{species}}"

# set this variable to true to download all images per observation
all_per_obs = True


# this variable controls the size of the downloaded images
# it can be "small", "medium", "large", or "square"
image_size = "large"


# max retries for each image before giving up
max_retries = 3

if all_per_obs:
    print(f"Image download size: {image_size}")
    number_of_images = 0
    json_file_path = f"../CommonNamesFilteredObservations/{taxon}/{common_name}.json"
    images_folder = f"TrainImages_{image_size}"  
    
    
    if os.path.exists(json_file_path):

        directory = f"../TrainImages_{image_size}/{taxon}"
        os.makedirs(directory, exist_ok=True)

        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                observations_list_of_dicts = data.get("results", [])
                obs_counter = 1
                for little_dict in observations_list_of_dicts:

                    if little_dict.get('photos'):

                        for image_counter, photo in enumerate(little_dict['photos']):
                            retries = 0
                            url = photo['url'].replace("square", image_size)
                            idd = little_dict['id']
                            successful_download = False
                            while retries < max_retries and not successful_download:
                                try:
                                    response = requests.get(url)
                                    if response.status_code == 200:
                                        
                                        image_path = os.path.join(directory, f"{common_name}_{obs_counter}_{idd}_{image_counter + 1}.jpg")
                                        with open(image_path, 'wb') as f_img:
                                            f_img.write(response.content)
                                            print(f"Downloaded observation {obs_counter} of {common_name}, example {image_counter + 1} with id {idd}")
                                            
                                        image_path_obj = Path(image_path)
                                        if image_path_obj.is_file():
                                            number_of_images += 1
                                            successful_download = True
                                             
                                    else:
                                        print(f"Failed to download observation {obs_counter} of {common_name}, example {image_counter + 1} (Status code: {response.status_code})")
                                        retries += 1
                                        time.sleep(0.1)
                                        
                                except Exception as e:
                                    print(f"Error downloading observation {obs_counter} of {common_name}, example {image_counter + 1}: {e}")
                                    retries += 1
                                    time.sleep(0.1)
                            if not successful_download:
                                print(f"Failed to download observation {obs_counter} of {common_name}, example {image_counter + 1} after {max_retries} retries.")

        
                        taxon_dir = f"../{images_folder}/{taxon}"          
                        observation_images = list(Path(taxon_dir).glob(f'{common_name}_*.jpg'))    
                        if observation_images:
                            obs_counter += 1  
                        
                        
                            
                    else:
                        idd = little_dict.get('id', 'unknown')
                        print(f"No photos available for {common_name} with id {idd}")

        except json.JSONDecodeError:
            print(f"Error: Failed to parse JSON in {json_file_path}")

        except UnicodeDecodeError:
            print(f"Error: Non-UTF-8 encoded data found in {json_file_path}")

        info_file = f"../AllDatasetInfo/{taxon}_Info.txt"
        os.makedirs(os.path.dirname(info_file), exist_ok=True)
        
        with open(info_file, "a", encoding='utf-8') as file:
            file.write(f"{common_name}:{number_of_images}\n")

    else:
        print(f"JSON file does not exist for {common_name}")




if not all_per_obs:

    print(f"Processing size: {image_size}")
    number_of_images = 0
    json_file_path = f"../CommonNamesFilteredObservations/{taxon}/{common_name}.json"

    if os.path.exists(json_file_path):
        
        directory = f"../TrainImages_{image_size}/{taxon}"
        os.makedirs(directory, exist_ok=True)

        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                observations_list_of_dicts = data.get("results", [])
                obs_counter = 1
                for little_dict in observations_list_of_dicts:

                    if little_dict.get('photos'):
                        retries = 0
                        url = little_dict['photos'][0]['url'].replace("square", "large")
                        idd = little_dict['id']
                        successful_download = False
                        while retries < max_retries and not successful_download:
                            try:
                                response = requests.get(url)
                                if response.status_code == 200:
                                    
                                    image_path = os.path.join(directory, f"{common_name}_{obs_counter}_{idd}.jpg")

                                    with open(image_path, 'wb') as f_img:
                                        f_img.write(response.content)
                                        print(f"Downloaded observation {obs_counter} of {common_name}, example {1} with id {idd}")
                                    image_path_obj = Path(image_path)
                                    if image_path_obj.is_file():
                                        number_of_images += 1
                                        successful_download = True
                                        obs_counter += 1       
                                else:
                                    
                                    print(f"Failed to download observation {obs_counter} of {common_name}, example {1} (Status code: {response.status_code})")
                                    retries += 1
                                    time.sleep(0.1)
                            except Exception as e:
                                print(f"Error downloading observation {obs_counter} of {common_name}, example {1}: {e}")
                                retries += 1
                                time.sleep(0.1)
                        if not successful_download:
                            print(f"Failed to download observation {obs_counter} of {common_name}, example {1} after {max_retries} retries.")


                    else:
                        idd = little_dict.get('id', 'unknown')
                        print(f"No photos available for {common_name} with id {idd}")

        except json.JSONDecodeError:
            print(f"Error: Failed to parse JSON in {json_file_path}")

        except UnicodeDecodeError:
            print(f"Error: Non-UTF-8 encoded data found in {json_file_path}")


        info_file = f"../AllDatasetInfo/{taxon}_Info.txt"
        os.makedirs(os.path.dirname(info_file), exist_ok=True)
        
        with open(info_file, "a", encoding='utf-8') as file:
            file.write(f"{common_name}:{number_of_images}\n")


    else:
        print(f"JSON file does not exist for {common_name}")




