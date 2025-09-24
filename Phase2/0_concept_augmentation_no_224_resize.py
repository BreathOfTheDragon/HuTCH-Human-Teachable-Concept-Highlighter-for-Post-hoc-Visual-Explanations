import os
from PIL import Image
import shutil



def augment_images(input_directory, output_directory):

    # delete the output directory if it exists
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Get all .jpg files in the input directory
    image_files = [f for f in os.listdir(input_directory) if f.endswith('.jpg')]

    for image_file in image_files:
        # Open the image
        image_path = os.path.join(input_directory, image_file)
        
        
        with Image.open(image_path) as img:
            # Save the original image
            img.save(os.path.join(output_directory, f"{os.path.splitext(image_file)[0]}_original.jpg"))

            # Rotate 90 degrees clockwise
            img_90 = img.rotate(-90, expand=True)
            img_90.save(os.path.join(output_directory, f"{os.path.splitext(image_file)[0]}_90cw.jpg"))

            # Rotate 180 degrees clockwise
            img_180 = img.rotate(-180, expand=True)
            img_180.save(os.path.join(output_directory, f"{os.path.splitext(image_file)[0]}_180cw.jpg"))

            # Rotate 270 degrees clockwise
            img_270 = img.rotate(-270, expand=True)
            img_270.save(os.path.join(output_directory, f"{os.path.splitext(image_file)[0]}_270cw.jpg"))

            # Flip vertically
            img_flip_vertical = img.transpose(Image.FLIP_TOP_BOTTOM)
            img_flip_vertical.save(os.path.join(output_directory, f"{os.path.splitext(image_file)[0]}_flipped_vertical.jpg"))

            # Flip horizontally
            img_flip_horizontal = img.transpose(Image.FLIP_LEFT_RIGHT)
            img_flip_horizontal.save(os.path.join(output_directory, f"{os.path.splitext(image_file)[0]}_flipped_horizontal.jpg"))


# concepts = ["BentAntenna", "Fur", "LongAntenna", "PinchWaist", "SegmentedAntenna", "Stripes"]
concepts = [ "Fur", "PinchWaist"]



if __name__ == "__main__":
    for concept in concepts:
        input_dir = f"./Concepts/Positive/Positive{concept}"
        output_dir = f"./AugmentedConcepts/Positive/Positive{concept}"
        augment_images(input_dir, output_dir)

        input_dir = f"./Concepts/Negative/Negative{concept}"
        output_dir = f"./AugmentedConcepts/Negative/Negative{concept}"
        augment_images(input_dir, output_dir)
