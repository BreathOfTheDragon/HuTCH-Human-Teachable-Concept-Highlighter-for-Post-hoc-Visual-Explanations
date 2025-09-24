# HuTCH-Human-Teachable-Concept-Highlighter-for-Post-hoc-Visual-Explanations

Please create a conda environment based on the provided requirements.txt or the HuTCH.yml file.


This project is structured in two phases.


The first phase consists of using the iNaturalist website to query the bee and wasp species of Colorado, downloading the related observations from iNaturalist, and training a CNN model (ResNet-152) from the downloaded images.


The second phase of the project is using the HuTCH framework to come up with the human teachable explanations to be compared to expert explanations and to be used in educational settings. 


# Phase 1:


The scripts are numbered and must be run in order.

### `1_GetColoradoCommonNamesFiltered.py`
* **Purpose**: Filters the initial species lists, keeping only those with a sufficient number of observations in Colorado.
* **Input**: `CommonNamesRaw/{taxon}.txt`
* **Output**: `CommonNamesFiltered/{taxon}_filtered.txt`

Note: This gives a long list of bee and wasp names to work with. We then manually keep the ones we are interested in, and discard the ones we are not.

---
### `2_GetColoradoCommonNamesFilteredObservations.py`
* **Purpose**: Downloads observation metadata (e.g., photo URLs) as JSON files from the iNaturalist API for the filtered species.
* **Input**: `CommonNamesFiltered/{taxon}_filtered.txt`
* **Output**: JSON files stored in `CommonNamesFilteredObservations/`

---
### `3_GetRanksFromCommonNamesFilteredObservations.py`
* **Purpose**: **(Optional)** An analysis script that extracts and saves taxonomic data (genus, family, etc.) from the JSON files. It is **not required** for the model training pipeline.
* **Input**: JSON files from `CommonNamesFilteredObservations/`
* **Output**: Text files containing rank information saved in various `...Rank` directories.

---
### `4_FastDownload...Images.py`
* **Purpose**: Downloads the actual JPG images for each species using the metadata from the JSON files.
* **Input**: JSON files from `CommonNamesFilteredObservations/`
* **Output**: Image files in `TrainImages_large/` and count summaries in `AllDatasetInfo/`

---
### `8_SortDatasetInfo.py`
* **Purpose**: A utility script to alphabetically sort the species and image counts in the info files.
* **Input**: `AllDatasetInfo/{taxon}_Info.txt`
* **Output**: The same file, but with its contents sorted.

---
### `9_SplitTrainTestDataV1.py`
* **Purpose**: Splits the dataset by moving a fraction of images (default 15%) into a separate directory for testing.
* **Input**: Images in `TrainImages_large/`
* **Output**: Moves files to `TestImages_large/` and creates corresponding info files in `TrainDatasetInfo/` and `TestDatasetInfo/`.

---
### `10_findDatasetNormalizationValues.py`
* **Purpose**: Calculates the dataset's mean and standard deviation, which are required for image normalization during training and testing.
* **Input**: Images from `TrainImages_large/`
* **Output**: Prints the values to the console. You must **manually copy these values** into scripts 11 and 12.

---
### `11_ModelTrainingWithCM.py`
* **Purpose**: Trains a ResNet-152 image classification model using the training data, validates it, and saves the trained model weights.
* **Input**: Images from `TrainImages_large/`
* **Output**: The trained model (`.pth` file) in `models/` and performance reports (confusion matrix, classification report) in `Reports/`.

---
### `12_ModelTestingWithCM.py`
* **Purpose**: Loads the trained model and evaluates its performance on the unseen test dataset.
* **Input**: The trained model from `models/` and images from `TestImages_large/`.
* **Output**: Final evaluation reports for the test set are saved in `Reports/`.
