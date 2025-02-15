import os
import cv2
from sklearn.metrics import accuracy_score
from ultralytics import YOLO

ALERT_CLASSES = ['knife', 'scissors']
CLASS_ID_MAP = {0: 'knife', 1: 'scissors'}  # Adjust the mapping based on your class IDs

def load_images_and_labels(image_dir, label_dir):
    images = []
    labels = []
    for img_file in os.listdir(image_dir):
        if img_file.endswith('.jpg') or img_file.endswith('.png'):
            img_path = os.path.join(image_dir, img_file)
            label_path = os.path.join(label_dir, img_file.replace('.jpg', '.txt').replace('.png', '.txt'))
            if os.path.exists(img_path) and os.path.exists(label_path):
                images.append(img_path)
                with open(label_path, 'r') as f:
                    label_lines = f.readlines()
                    labels.append([int(line.split()[0]) for line in label_lines])
            else:
                print(f"Warning: Image or label file does not exist for {img_file}")
    return images, labels

def evaluate_model(model_path, image_dir, label_dir):
    model = YOLO(model_path)
    images, labels = load_images_and_labels(image_dir, label_dir)
    predictions = []
    filtered_labels = []
    label_counts = {cls: 0 for cls in ALERT_CLASSES}
    prediction_counts = {cls: 0 for cls in ALERT_CLASSES}

    for image_path, label_list in zip(images, labels):
        if not os.path.exists(image_path):
            print(f"Warning: Image file does not exist {image_path}")
            continue
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Unable to read image file {image_path}")
            continue
        results = model(image)
        if results and results[0].boxes:
            for cls in results[0].boxes.cls.cpu().numpy():
                pred_label = results[0].names[int(cls)]
                if pred_label in ALERT_CLASSES:
                    predictions.append(pred_label)
                    prediction_counts[pred_label] += 1
                    filtered_labels.append(pred_label)
        for class_id in label_list:
            if CLASS_ID_MAP[class_id] in ALERT_CLASSES:
                label_counts[CLASS_ID_MAP[class_id]] += 1

    if not predictions or not filtered_labels:
        return float('nan')

    accuracy = accuracy_score(filtered_labels, predictions)
    print(f"Label counts: {label_counts}")
    print(f"Prediction counts: {prediction_counts}")
    return accuracy

def main():
    image_dir = r'F:\Faculdade\FIAP-VisionGuard-Grupo-15\configs\coco_cutting_yolo_val\images\val'
    label_dir = r'F:\Faculdade\FIAP-VisionGuard-Grupo-15\configs\coco_cutting_yolo_val\labels\val'
    model_path = r'F:\Faculdade\FIAP-VisionGuard-Grupo-15\yolov8m.pt'
    evaluate_model(model_path, image_dir, label_dir)

if __name__ == "__main__":
    main()