import requests
import os
import json
import time


# This code comes up with CommonNamesFiltered lists. Basically, a list of all common names that were observed at least a certain number of times in Colorado 
# (defined by minimum_num_of_observations)
# In this code, 50 is the minimum required obserevation count for a common name to be considered as Colorado Native


taxons = ["bee", "wasp"]



wing_dudes = {
    "bee": 630955,
    "wasp": 52747,
}

place_id = 34  # colorado is 34 apparently

obs_per_page = 50
num_of_pages = 1

# min number of observation of a species in Colorado to be considered a native species:

minimum_num_of_observations = obs_per_page * num_of_pages

print(f"Minimum number of observations per species is set to: {minimum_num_of_observations}")


def get_observations(common_name, per_page=obs_per_page, max_retries=5):
    url = "https://api.inaturalist.org/v1/observations"
    observations = []
    page = 1
    while True:
        params = {
            "taxon_name": common_name,
            "place_id": place_id,
            "per_page": per_page,
            "page": page
        }
        print(f"Working on {common_name} (Page {page})")
        response = None
        retries = 0

        while retries < max_retries:
            try:
                response = requests.get(url, params=params)
                if response.status_code == 200:
                    break
                else:
                    print(f"Failed to retrieve page {page} for {common_name} on try {retries + 1}, status code {response.status_code}")
                    retries += 1
                    time.sleep(2)
            except Exception as e:
                print(f"Error retrieving page {page} for {common_name}: {e}")
                retries += 1
                time.sleep(1)

        if response is None or retries == max_retries:
            print(f"Max retries reached for page {page}, skipping to next page.")
            break

        data = response.json()
        results = data.get("results", [])
        if not results:
            print(f"No more results found for {common_name} after page {page}.")
            break

        observations.extend(results)
        page += 1

        if page > num_of_pages:
            break
        
    time.sleep(2)        
    return observations


def not_enough_observations(common_name, observations, minimum_num_of_observations=minimum_num_of_observations) -> bool:
    num_of_observations = len(observations)
    if num_of_observations < minimum_num_of_observations:
        print(f"Error! There are NOT at least {minimum_num_of_observations} observations for the --- {common_name} --- species in Colorado!"
              " removing from list")
        print()
        return True
    else:
        print(f"Success! There are at least {minimum_num_of_observations} observations for the --- {common_name} --- species in Colorado!"
              " keeping in the list")
        print()
        return False


def filter_taxa(taxon):
    filename = f'./CommonNamesRaw/{taxon}.txt'
    all_species = set()
    
    # if file doesnt exist, create it
    if not os.path.exists(filename):
        with open(filename, 'w') as f:
                pass
        
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            for line in f:
                all_species.add(line.strip())


    all_species_sorted = sorted(all_species)
    filtered_all_species_sorted = []

    for common_name in all_species_sorted:
        observations = get_observations(common_name)
        if not not_enough_observations(common_name, observations):
            filtered_all_species_sorted.append(common_name)


    filtered_species_dir = './CommonNamesFiltered'
    os.makedirs(filtered_species_dir, exist_ok=True)
    filename_filtered = f"{filtered_species_dir}/{taxon}_filtered.txt"
            
    with open(f"{filename_filtered}", 'w') as f:
        for name in filtered_all_species_sorted:
            f.write(f"{name}\n")

    print("**********************************************************************************************")
    print(f"Filtered, Sorted and Saved {len(filtered_all_species_sorted)} species for {taxon} in {filename}.")
    print("**********************************************************************************************")


for taxon in taxons:
    filter_taxa(taxon)
