import cv2
from torchvision import transforms

def prepare_image(image_path):
    """
    Load and preprocess the image for object detection.
    """
    # Load image using OpenCV
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Unable to load image: {image_path}")
    
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Convert image to tensor
    transformations = transforms.Compose([transforms.ToTensor()])
    image_tensor = transformations(image_rgb)

    return image_tensor, image_rgb  # Return tensor and RGB image for later use


def annotate_image(image, detection_results, threshold=0.5):
    """
    Draw bounding boxes and labels on the image.
    """
    boxes = detection_results["boxes"]
    scores = detection_results["scores"]
    labels = detection_results["labels"]

    for i in range(len(boxes)):
        if scores[i] >= threshold:
            # Extract bounding box coordinates
            x1, y1, x2, y2 = map(int, boxes[i])
            
            # Draw rectangle
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Prepare label with confidence score
            label = f"{labels[i].item()} ({scores[i]:.2f})"
            
            # Add label text
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return image
