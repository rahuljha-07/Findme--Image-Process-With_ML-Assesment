import os
import cv2
from model_loader import load_pretrained_model
from image_processor import prepare_image, annotate_image
import torch

def process_images(input_folder, output_folder, detection_model):
    """
    Process images from the input folder, detect objects, and save annotated images.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        image_path = os.path.join(input_folder, filename)
        if not filename.lower().endswith(('.jpg', '.png', '.jpeg')):
            continue

        try:
            # Prepare the image
            image_tensor, original_image = prepare_image(image_path)

            # Perform object detection
            with torch.no_grad():
                detections = detection_model([image_tensor])[0]

            # Annotate the image with detections
            annotated_image = annotate_image(original_image, detections)

            # Save the annotated image
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
            print(f"Processed and saved: {output_path}")

        except Exception as error:
            print(f"Error processing {filename}: {error}")


def main():
    input_directory = "../images"
    output_directory = "../output"

    print("Loading the detection model...")
    detection_model = load_pretrained_model()

    print("Processing images in the input directory...")
    process_images(input_directory, output_directory, detection_model)

    print("Processing complete. Check the output directory for results.")


if __name__ == "__main__":
    main()
