import os
from PIL import Image
import shutil

def augment_images(input_directory, output_directory):


    if os.path.exists(output_directory):
        shutil.rmtree(output_directory)
    os.makedirs(output_directory, exist_ok=True)


    image_files = [f for f in os.listdir(input_directory) if f.lower().endswith(('.jpg', '.png', '.jpeg')) and not f.startswith('.')]
    
    for image_file in image_files:
        
        
        
        image_path = os.path.join(input_directory, image_file)
        with Image.open(image_path) as img:
            
            size = (224,224)

            img.thumbnail(size, Image.Resampling.LANCZOS)
                            
            scaling_factors = [1/4, 1/3, 1/2.5, 1/2, 1/1.5, 1]

            for scale in scaling_factors:


                width, height = img.size
                new_width = int(width * scale)
                new_height = int(height * scale)

                # Rotate 0 degrees clockwise
                scaled_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                scaled_img.save(os.path.join(output_directory, f"{os.path.splitext(image_file)[0]}_scaled_{int(scale*100)}.jpg"))


                # Rotate 90 degrees clockwise
                img_90 = scaled_img.rotate(-90, expand=True)
                img_90.save(os.path.join(output_directory, f"{os.path.splitext(image_file)[0]}_scaled_{int(scale*100)}_90cw.jpg"))

                # Rotate 180 degrees clockwise
                img_180 = scaled_img.rotate(-180, expand=True)
                img_180.save(os.path.join(output_directory, f"{os.path.splitext(image_file)[0]}_scaled_{int(scale*100)}_180cw.jpg"))

                # Rotate 270 degrees clockwise
                img_270 = scaled_img.rotate(-270, expand=True)
                img_270.save(os.path.join(output_directory, f"{os.path.splitext(image_file)[0]}_scaled_{int(scale*100)}_270cw.jpg"))

                # Flip vertically
                img_flip_vertical = scaled_img.transpose(Image.FLIP_TOP_BOTTOM)
                img_flip_vertical.save(os.path.join(output_directory, f"{os.path.splitext(image_file)[0]}_scaled_{int(scale*100)}_flipped_vertical.jpg"))

                # Flip vertically and Rotate 90 degrees clockwise
                img_flip_vertical_90 = img_flip_vertical.rotate(-90, expand=True)
                img_flip_vertical_90.save(os.path.join(output_directory, f"{os.path.splitext(image_file)[0]}_scaled_{int(scale*100)}_flipped_vertical_90cw.jpg"))


                # Flip vertically and Rotate 180 degrees clockwise
                img_flip_vertical_180 = img_flip_vertical.rotate(-180, expand=True)
                img_flip_vertical_180.save(os.path.join(output_directory, f"{os.path.splitext(image_file)[0]}_scaled_{int(scale*100)}_flipped_vertical_180cw.jpg"))

                
                # Flip vertically and Rotate 270 degrees clockwise
                img_flip_vertical_270 = img_flip_vertical.rotate(-270, expand=True)
                img_flip_vertical_270.save(os.path.join(output_directory, f"{os.path.splitext(image_file)[0]}_scaled_{int(scale*100)}_flipped_vertical_270cw.jpg"))






concepts = ["Fur", "PinchWaist"]

if __name__ == "__main__":
    for concept in concepts:
        input_dir = f"./ConceptsFinal/Positive/Positive{concept}"
        output_dir = f"./AugmentedConceptsFinal/Positive/Positive{concept}"
        augment_images(input_dir, output_dir)

        input_dir = f"./ConceptsFinal/Negative/Negative{concept}"
        output_dir = f"./AugmentedConceptsFinal/Negative/Negative{concept}"
        augment_images(input_dir, output_dir)
