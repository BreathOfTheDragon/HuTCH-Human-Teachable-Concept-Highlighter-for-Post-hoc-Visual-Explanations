# HuTCH: Human-Teachable Concept Highlighter for Post-hoc Visual Explanations

Create a conda environment based on the provided `requirements.txt` or the `HuTCH.yml` file.

This project is structured in two phases.

The first phase consists of using the iNaturalist website to query the bee and wasp species of Colorado, downloading the related observations from iNaturalist, and training a CNN model (ResNet-152) from the downloaded images.

The second phase of the project is using the HuTCH framework to come up with the human-teachable explanations to be compared to expert explanations and to be used in educational settings. 

# Phase 1:

The scripts are numbered and must be run in order.

### `1_GetColoradoCommonNamesFiltered.py`
* **Purpose**: Filters the initial species lists, keeping only those with a sufficient number of observations in Colorado.
* **Input**: `CommonNamesRaw/{taxon}.txt`
* **Output**: `CommonNamesFiltered/{taxon}_filtered.txt`
* **Note**: This script provides a long list of bee and wasp names. Manually edit the output list to keep only the desired species.

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

NOTE: We don't run these scripts manually, instead we use the `ImageDownloadParallelScript.sh` script to create a python file for each of the bee or wasp groups so that we can run them all in parallel and save on downloading time.

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
* **Output**: Prints the values to the console. **Manually copy these values** into scripts 11 and 12.

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


# Phase 2:

The scripts are numbered and must be run in order.

**Note**: The following folders must be created and populated before running this phase: `ConceptsFinal`, `highlighted_images_by_expert`, `images_to_highlight_unclean`, and `TestImages`.

## Required User-Provided Folders
The entire workflow relies on several key folders that must be created and populated with data. These folders provide the initial inputs for training the concepts and evaluating the results.

### `ConceptsFinal`
* **What it is**: This folder is the **knowledge base for teaching the model high-level concepts**. It must contain `Positive` and `Negative` subdirectories. Inside these, create a folder for each concept (e.g., `PositiveFur`, `NegativeFur`) and fill it with example images.
* **Why it's needed**: These images are the starting point for Phase 2. Script `2_create_the_classifier.py` uses these positive and negative examples to learn what a concept like "Fur" looks like and generate the Concept Activation Vectors (CAVs) that are central to the project.

### `highlighted_images_by_expert`
* **What it is**: This folder contains the **ground truth for evaluation**. For each test image, an expert must manually create a corresponding image that highlights the exact region of a concept. These are typically PNGs with transparent backgrounds where only the concept (e.g., the furry part of a bee) is visible.
* **Why it's needed**: This is the benchmark used in the final evaluation scripts (`6`, `11`, etc.) to judge the performance of the computer-generated highlights. The scripts compare the model's output against these expert images to calculate quantitative metrics like IoU and Dice scores. Without this ground truth, it is not possible to measure how well the system is working.

### `images_to_highlight_unclean`
* **What it is**: This folder holds the raw, original images for the model to **analyze and explain**. These are the target images for the highlighting process.
* **Why it's needed**: This is the input for the explanation pipeline that begins in Phase 2. Script `3` takes these "unclean" images (with backgrounds) and segments the main object out. The final output of the project (in script `5`) is a version of these images with the relevant concepts highlighted.

### `TestImages`
* **What it is**: This folder acts as a **label map** for the model's predictions. It should contain subdirectories with the names of the classes the original model was trained on (e.g., a folder named `bee` and another named `wasp`). The content of these folders doesn't matter, only their names.
* **Why it's needed**: In Phase 2, script `5_show_highlights.py` first predicts the class of an image (e.g., "bee"). It then uses the folder names in `TestImages` to map the prediction index (e.g., 0) to a class name ("bee"). This class name is then used to decide which concept to highlight (e.g., "fur" for bees).


---

### `1_option1...` OR `1_option2_..._augmentation...`
* **Purpose**: Augments the positive and negative concept images by applying a series of rotations, scales, and flips to increase the dataset size.
* **Note**: These are alternatives. Choose **`option1`** for fewer augmentations (8 per image) or **`option2`** for a more comprehensive set (24 per image).
* **Input**: Manually created concept images in `ConceptsFinal/`.
* **Output**: Augmented images in `AugmentedConceptsFinal/`.

---
### `2_create_the_classifier.py`
* **Purpose**: Creates the **Concept Activation Vectors (CAVs)**. It passes the augmented concept images through the pre-trained model to get their activations, then trains a simple linear classifier to find the vector that separates the positive and negative examples for each concept.
* **Input**: Augmented images from `AugmentedConceptsFinal/` and the model from `models/`.
* **Output**:
    * Activations for each concept in the `activations/` directory.
    * The final CAVs, intercepts, and scalers in the `cavs/` directory.

---
### `3_option1...` OR `3_option2_..._clean_the_input_images...`
* **Purpose**: Takes raw images and "cleans" them by using a Mask R-CNN model to detect the main object, crop it out, and save it on a transparent background.
* **Note**: These are alternatives. Choose **`option1`** to have the final cleaned images resized to `224x224` or **`option2`** to keep them at their original cropped size.
* **Input**: Raw images from `images_to_highlight_unclean/`.
* **Output**: Cleaned PNGs with transparent backgrounds in `images_to_highlight_for_computer_clean/` and `images_to_highlight_for_expert_clean/`.

---
### `4_1_create_sub_images_multi.py`
* **Purpose**: Generates many variations of each cleaned image. It creates "blackened" sub-images by isolating small patches on a grid and "segmented" sub-images by breaking the object into its component parts using Mask R-CNN.
* **Input**: Cleaned images from `images_to_highlight_for_computer_clean/`.
* **Output**: A large number of sub-images in the `blackened_images/` and `segmented_images/` directories.

---
### `4_2_create_super_images_multi.py`
* **Purpose**: Takes the sub-images and creates "super-images" by combining them. In its current form (`n=1`), this script primarily copies the sub-images into new directories in preparation for the next step.
* **Input**: Sub-images from `blackened_images/` and `segmented_images/`.
* **Output**: Reorganized images in `super_blackened_images/` and `super_segmented_images/`.

---
### `4_3_calculate_activations_of_super_images.py`
* **Purpose**: Calculates the model activations for every "super-image" created in the previous step.
* **Input**: Images from `super_blackened_images/` and `super_segmented_images/`.
* **Output**: Pickled dictionaries containing the activations for each image, saved in `name_activation_dict_blackened/` and `name_activation_dict_segmented/`.

---
### `4_AllinOne_create_sub_images_and_calculate_activations_of_them.py`
* **Purpose**: **(Optional)** This script combines the functionality of `4_1`, `4_2`, and `4_3` into a single file. It is a less modular alternative to running those three scripts sequentially.
* **Input**: Cleaned images from `images_to_highlight_for_computer_clean/`.
* **Output**: Final activation dictionaries in `name_activation_dict_blackened/` and `name_activation_dict_segmented/`.

---
### `5_show_highlights.py`
* **Purpose**: The final step. It uses the CAVs (from script 2) to score the activations of each sub-image (from script 4.3). It then identifies the top-scoring sub-images that best represent a concept (e.g., "fur") and overlays them to create a final visualization that "highlights" the concept on the original image.
* **Input**:
    * CAVs from `cavs/`.
    * Activation dictionaries from `name_activation_dict.../`.
    * Super-images from `super_..._images/`.
* **Output**: The final highlighted images saved in `highlighted_images_by_computer/`.


---
#### `6_calculate_overlap_and_metrics.py`
* **Purpose**: Compares the computer-generated highlights (both `blackened` and `segmented` versions) from Phase 2 against the manually created expert highlights. It calculates the **Intersection over Union (IoU)** and **Dice** similarity scores for each image.
* **Input**:
    * `highlighted_images_by_computer/`
    * `highlighted_images_by_expert/`
* **Output**:
    * CSVs with per-image scores (`iou_dice_results_blackened.csv`, `..._segmented.csv`).
    * Visualizations of the overlap saved in `highlighted_images_overlap/`.

---
#### `7_calcualte_mean_std_metrics.py`
* **Purpose**: Reads the per-image scores from the previous step and calculates aggregate statistics (mean, standard deviation, and 95% confidence interval) for each highlight method.
* **Input**: The CSV files generated by script `6`.
* **Output**: New CSVs with the aggregate stats (`iou_dice_stats_blackened.csv`, `..._segmented.csv`).

### Saliency-based Method Generation & Evaluation

---
#### `8_create_grad_saliency_map.py`
* **Purpose**: Generates gradient-based saliency maps. This is an alternative highlighting method that identifies which pixels in an image were most important for the model's classification decision.
* **Input**: Cleaned images from `images_to_highlight_for_expert_clean/` and the trained model from `models/`.
* **Output**: Saliency map images saved in various formats within the `saliency_images_by_model/` directory.

---
#### `9_1_create_sub_saliency_images.py` & `9_2_create_super_saliency_multi.py`
* **Purpose**: These scripts process the saliency maps. **`9_1`** breaks the maps into smaller patches. **`9_2`** reorganizes these patches into a "super-image" format for the next step.
* **Input**: Saliency maps from `saliency_images_by_model/`.
* **Output**: Processed sub-images saved in `super_blackened_saliency_images/`.

---
#### `10_option1_show_saliency_highlights_average.py`
* **Purpose**: Creates the final computer-generated highlight for the saliency method. It scores each saliency patch based on pixel intensity (brighter is more important) and combines the top-scoring patches into a single image.
* **Input**: Images from `super_blackened_saliency_images/`.
* **Output**: Final saliency highlights saved in `highlighted_saliency_images_by_computer/`.

---
#### `11_calculate_overlap_and_metrics.py`
* **Purpose**: The same as script `6`, but this time it compares the final **saliency-based** highlights against the expert highlights, calculating IoU and Dice scores.
* **Input**:
    * `highlighted_saliency_images_by_computer/`
    * `highlighted_images_by_expert/`
* **Output**: A CSV with per-image scores (`iou_dice_results_saliency.csv`) and overlap visualizations.

---
#### `12_calcualte_mean_std_metrics.py`
* **Purpose**: The same as script `7`, but calculates the aggregate statistics for the **saliency-based** method's scores.
* **Input**: The CSV file generated by script `11`.
* **Output**: A new CSV with aggregate stats (`iou_dice_stats_saliency.csv`).

### Final Analysis and Visualization

---
#### `13_statistical_tests.py`
* **Purpose**: Performs statistical analysis (**t-tests** and **ANOVA**) to determine if the performance differences between the three methods (Saliency, Blackened-CAV, Segmented-CAV) are statistically significant.
* **Input**: The per-image results CSVs generated by scripts `6` and `11`.
* **Output**: Prints statistical results (t-statistic, p-value) to the console.

---
#### `14_box_plots.py`
* **Purpose**: Creates the final visualizations for the project. It generates error bar plots to visually compare the mean IoU and Dice scores (with 95% confidence intervals) of the different highlighting methods.
* **Input**: The aggregate statistics CSVs generated by scripts `7` and `12`.
* **Output**: Two image files, `combined_iou.png` and `combined_dice.png`.
