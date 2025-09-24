#!/bin/bash

mkdir -p ./AutoPython

while IFS= read -r species; do
    echo "${species}"
    species_filename="${species}"
    output_file="AutoPython/Download_${species_filename}.py"

    species_escaped=$(printf '%s\n' "$species" | sed 's/[\/&]/\\&/g')

    sed "s/{{species}}/$species_escaped/" 4_FastDownloadBeeImages.py > "$output_file"

    (
        cd AutoPython
        /s/babbage/b/nobackup/nblancha/merry/conda/envs/ERFAN-ENV/bin/python3 "$(basename "$output_file")"
    ) &
done < ./CommonNamesFiltered/bee_filtered.txt

wait

echo "Done"




while IFS= read -r species; do
    echo "${species}"
    species_filename="${species}"
    output_file="AutoPython/Download_${species_filename}.py"

    species_escaped=$(printf '%s\n' "$species" | sed 's/[\/&]/\\&/g')

    sed "s/{{species}}/$species_escaped/" 4_FastDownloadWaspImages.py > "$output_file"

    (
        cd AutoPython
        /s/babbage/b/nobackup/nblancha/merry/conda/envs/ERFAN-ENV/bin/python3 "$(basename "$output_file")"
    ) &
done < ./CommonNamesFiltered/wasp_filtered.txt

wait

echo "Done"