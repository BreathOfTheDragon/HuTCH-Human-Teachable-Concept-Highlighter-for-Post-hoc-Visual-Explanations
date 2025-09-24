import json
import os


taxons = ["bee", "fly", "wasp"]


big_num = 0



for taxon in taxons:
    
    taxon_dict_Ranks = dict()
    taxon_dict_Species = dict()
    taxon_dict_Lisa = dict()
    taxon_dict_Lisa_with_colon = dict()
    
    
    ranks = set()  # a set of all ranks [species, genus, family, order] etc. 

    specific_ranks = set()  # This set will contain specific ranks, like family:Megachilidae or genus:Agapostemon

    filename = f'./CommonNamesFiltered/{taxon}_filtered.txt'
    common_names = set()
    
    with open(filename, 'r') as f:
        for line in f:
            common_names.add(line.strip())
        
        
    for common_name in common_names:
        
        common_name_species = set()
        common_name_Lisa = set()
        common_name_Lisa_with_colon = set()
        common_name_rank = set()
        
        with open(f'./CommonNamesFilteredObservations/{taxon}/{common_name}.json') as f:
            data = json.load(f)



        for i in range(len(data['results'])):

            ranks.add(data['results'][i]['taxon']['rank'])
            common_name_rank.add(f"{data['results'][i]['taxon']['rank']}:{data['results'][i]['taxon']['name']}")
            if data['results'][i]['taxon']['rank'] == "species":
                common_name_species.add(f"{data['results'][i]['taxon']['name']}")
                
            if data['results'][i]['taxon']['rank'] == "species" or data['results'][i]['taxon']['rank'] == "genus" or data['results'][i]['taxon']['rank'] == "family":
                common_name_Lisa.add(f"{data['results'][i]['taxon']['name']}")
            
            if data['results'][i]['taxon']['rank'] == "species" or data['results'][i]['taxon']['rank'] == "genus" or data['results'][i]['taxon']['rank'] == "family":
                common_name_Lisa_with_colon.add(f"{data['results'][i]['taxon']['rank']}:{data['results'][i]['taxon']['name']}")
            
               
            specific_ranks.add(f"{data['results'][i]['taxon']['rank']}:{data['results'][i]['taxon']['name']}")
            print(f"{data['results'][i]['taxon']['rank']}:{data['results'][i]['taxon']['name']}")
            big_num+=1
        

        
        list_of_common_name_rank= list(common_name_rank)
        sorted_list_of_common_name_rank = sorted(list_of_common_name_rank)
        taxon_dict_Ranks[common_name] = sorted_list_of_common_name_rank
        
        
        list_of_common_name_species= list(common_name_species)
        sorted_list_of_common_name_species = sorted(list_of_common_name_species)
        taxon_dict_Species[common_name] = sorted_list_of_common_name_species


        list_of_common_name_Lisa= list(common_name_Lisa)
        sorted_list_of_common_name_Lisa = sorted(list_of_common_name_Lisa)
        taxon_dict_Lisa[common_name] = sorted_list_of_common_name_Lisa


        list_of_common_name_Lisa_with_colon= list(common_name_Lisa_with_colon)
        sorted_list_of_common_name_Lisa_with_colon = sorted(list_of_common_name_Lisa_with_colon)
        taxon_dict_Lisa_with_colon[common_name] = sorted_list_of_common_name_Lisa_with_colon



    list_of_specific_ranks = list(specific_ranks)
    sorted_list_of_specific_ranks = sorted(list_of_specific_ranks)


    


    common_names = taxon_dict_Ranks.keys()    
    sorted_common_names = sorted(common_names)
    
    
    
    
    directory = f"./CommonNamesFilteredAllRanks"
    os.makedirs(directory, exist_ok=True)
    the_file = f"{directory}/{taxon}.txt"
    with open(the_file, 'w') as file:
        file.write('')
    
    for common_name in sorted_common_names:
        the_file = f"{directory}/{taxon}.txt"
        with open(the_file, 'a') as file:
            file.write(f"---------- {common_name} ----------\n")
            for line in taxon_dict_Ranks[common_name]:
                file.write(f"{line}\n")   

    
    
    
    directory = f"./CommonNamesFilteredSpeciesRank"
    os.makedirs(directory, exist_ok=True)
    the_file = f"{directory}/{taxon}.txt"
    with open(the_file, 'w') as file:
        file.write('')
    
    for common_name in sorted_common_names:
        the_file = f"{directory}/{taxon}.txt"
        with open(the_file, 'a') as file:
            file.write(f"---------- {common_name} ----------\n")
            for line in taxon_dict_Species[common_name]:
                file.write(f"{line}\n")   


    
    directory = f"./CommonNamesFilteredLisaRank"
    os.makedirs(directory, exist_ok=True)
    the_file = f"{directory}/{taxon}.txt"
    with open(the_file, 'w') as file:
        file.write('')
    
    for common_name in sorted_common_names:
        the_file = f"{directory}/{taxon}.txt"
        with open(the_file, 'a') as file:
            file.write(f"---------- {common_name} ----------\n")
            for line in taxon_dict_Lisa[common_name]:
                file.write(f"{line}\n")   




    directory = f"./CommonNamesFilteredLisaWithColonRank"
    os.makedirs(directory, exist_ok=True)
    the_file = f"{directory}/{taxon}.txt"
    with open(the_file, 'w') as file:
        file.write('')
    
    for common_name in sorted_common_names:
        the_file = f"{directory}/{taxon}.txt"
        with open(the_file, 'a') as file:
            file.write(f"---------- {common_name} ----------\n")
            for line in taxon_dict_Lisa_with_colon[common_name]:
                file.write(f"{line}\n")   





    directory = f"./AllRanks"
    os.makedirs(directory, exist_ok=True)
    the_file = f"{directory}/{taxon}.txt"
    with open(the_file, 'w') as file:
        for line in sorted_list_of_specific_ranks:
            file.write(f"{line}\n")
            
