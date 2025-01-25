import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.datapoints")
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.transforms.v2")

def detect(img, detector_model):

    result = detector_model(img, verbose=False)[0]  # predict on an image

    # Get boxes and process each person detection
    boxes = result.boxes  # Boxes object for bounding box outputs

    coords=[]
    for box in boxes:
        # Check if detection is a person (class 0 in YOLO)
        if box.cls == 0:
            # Get box coordinates and crop image
            x1, y1, x2, y2 = box.xyxy[0]  # Get coordinates in (x1, y1, x2, y2) format
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # Convert to integers
            coords.append((x1, y1, x2, y2))
            
    return coords


