import os
from ultralytics import YOLO

def validate_model():
    # Define paths
    model_path = r'F:\Faculdade\FIAP-VisionGuard-Grupo-15\yolov8m.pt'
    data_yaml = r'F:\Faculdade\FIAP-VisionGuard-Grupo-15\configs\data.yaml'

    # Load the model
    model = YOLO(model_path)

    # Perform validation
    results = model.val(data=data_yaml)

    # Print the results
    print(results)

if __name__ == '__main__':
    validate_model()