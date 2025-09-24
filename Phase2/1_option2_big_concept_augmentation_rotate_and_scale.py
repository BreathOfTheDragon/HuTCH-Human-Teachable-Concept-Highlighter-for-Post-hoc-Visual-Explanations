import os
from PIL import Image
import shutil

def augment_images(input_directory, output_directory):

    if os.path.exists(output_directory):
        shutil.rmtree(output_directory)
    os.makedirs(output_directory, exist_ok=True)


    image_files = [
        f for f in os.listdir(input_directory)
        if f.lower().endswith(('.jpg', '.png', '.jpeg')) and not f.startswith('.')
    ]
    

    scaling_factors = [1/4, 1/3, 1/2.5, 1/2, 1/1.5, 1]
    

    rotations = [
        (0, "OG"), 
        (15, "15cw"),
        (30, "30cw"),
        (45, "45cw"), 
        (60, "60cw"),
        (75, "75cw"),
        (90, "90cw"),
        (105, "105cw"),
        (120, "120cw"),
        (135, "135cw"),
        (150, "150cw"),
        (165, "165cw"),
        (180, "180cw"),
        (195, "195cw"),
        (210, "210cw"),
        (225, "225cw"),
        (240, "240cw"),
        (255, "255cw"),
        (270, "270cw"),
        (285, "285cw"),
        (300, "300cw"),
        (315, "315cw"),
        (330, "330cw"),
        (345, "345cw")
    ]
    
    for image_file in image_files:
        image_path = os.path.join(input_directory, image_file)
        with Image.open(image_path) as img:

            target_size = (224, 224)
            img.thumbnail(target_size, Image.Resampling.LANCZOS)
            

            for scale in scaling_factors:
                width, height = img.size
                new_width = int(width * scale)
                new_height = int(height * scale)
                scaled_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                
                # --- 1 rotations to the scaled image ---
                for angle, suffix in rotations:

                    rotated_img = scaled_img.rotate(-angle, expand=True)
                    
                    base_name = os.path.splitext(image_file)[0]
                    filename = f"{base_name}_scaled_{int(scale*100)}_{suffix}.jpg"
                    rotated_img.save(os.path.join(output_directory, filename))
                
                # --- 2 rotations to the flipped and scaled image ---
                
                flipped_img = scaled_img.transpose(Image.FLIP_TOP_BOTTOM)
                
                for angle, suffix in rotations:
                    
                    rotated__flipped_img = flipped_img.rotate(-angle, expand=True)
                    
                    base_name = os.path.splitext(image_file)[0]
                    filename = f"{base_name}_flipped_scaled_{int(scale*100)}_{suffix}.jpg"
                    rotated__flipped_img.save(os.path.join(output_directory, filename))
                
                

if __name__ == "__main__":
    
    concepts = ["Fur", "PinchWaist"]

    for concept in concepts:

        input_dir = f"./ConceptsFinal/Positive/Positive{concept}"
        output_dir = f"./AugmentedConceptsFinal/Positive/Positive{concept}"
        augment_images(input_dir, output_dir)


        input_dir = f"./ConceptsFinal/Negative/Negative{concept}"
        output_dir = f"./AugmentedConceptsFinal/Negative/Negative{concept}"
        augment_images(input_dir, output_dir)
