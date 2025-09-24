import requests
import os
import json
import time


# This file comes up with the CommonNamesFilteredObservations. Basically downloads the json files of the observations, provided the list of common names in CommonNamesFiltered


# taxons = ["bee", "fly", "wasp"]

taxons = ["bee", "wasp"]

# taxons = ["bee"]



wing_dudes = {
    "bee": 630955,
    "wasp": 52747,
    "fly": 47822
}



# No need for place_id, as we already filtered for colorado species and made sure there have been a minimum number of observations
# place_id = 34  # colorado is 34 apparently


# change obs_per_page param to download observations per page
# change num_of_pages for number of pages


obs_per_page = 200
# num_of_pages = 50
num_of_pages = 50

total_observations = obs_per_page * num_of_pages

print(f"Total number of observations per species will be : {total_observations}")

    
    
def get_observations(common_name, per_page = obs_per_page, max_retries=5):
    url = "https://api.inaturalist.org/v1/observations"
    observations = []
    page = 1
    
    
    while True:
        params = {
            "taxon_name": common_name,
            #"place_id": place_id,
            "per_page": per_page,
            "page": page
        }
        print(f"Working on {common_name}")
        print(f"Page number {page}")
        response = None
        retries = 0

        while retries < max_retries:
            try:
                response = requests.get(url, params=params)
                if response.status_code == 200:
                    break
                else:
                    print(
                        f"Failed to retrieve page {page} for {common_name} on try {retries + 1}, status code {response.status_code}")
                    retries += 1
                    time.sleep(3)
            except Exception as e:
                print(f"Error retrieving page {page} for {common_name}: {e}")
                retries += 1
                time.sleep(3)

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
        
    return observations


for taxon in taxons:
    with open(f'./CommonNamesFiltered/{taxon}_filtered.txt', 'r') as f:
        common_names = f.read().splitlines()


    for common_name in common_names:
        observations = get_observations(common_name)
        # Species names under taxon names: (for example fervid nomad bee under bee)
        # download_folder = f"./{taxon}/{common_name}"

        # All jsons in the taxon name:
        download_folder = f"./CommonNamesFilteredObservations/{taxon}"
        os.makedirs(download_folder, exist_ok=True)

        try:
            with open(f'{download_folder}/{common_name}.json', 'w') as f:
                json.dump({"results": observations}, f)
            
        except:
            pass
