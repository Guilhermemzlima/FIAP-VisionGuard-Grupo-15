import fiftyone as fo
import fiftyone.zoo as foz

# Define the categories of interest
classes_of_interest = ["knife", "scissors"]

# Load the COCO 2017 training dataset
train_dataset = foz.load_zoo_dataset(
    "coco-2017",
    split="train",
    label_types=["detections"],
    classes=classes_of_interest,
    max_samples=500,  # Optional: limit the number of samples for testing
)

# Load the COCO 2017 validation dataset
val_dataset = foz.load_zoo_dataset(
    "coco-2017",
    split="validation",
    label_types=["detections"],
    classes=classes_of_interest,
    max_samples=100,  # Optional: limit the number of samples for testing
)

# Export the training dataset to YOLO format
train_export_dir = "./configs/coco_cutting_yolo_train"
train_dataset.export(
    export_dir=train_export_dir,
    dataset_type=fo.types.YOLOv5Dataset,
    classes=classes_of_interest,
    export_media=True,  # Copy the images to the export directory
)

# Export the validation dataset to YOLO format
val_export_dir = "./configs/coco_cutting_yolo_val"
val_dataset.export(
    export_dir=val_export_dir,
    dataset_type=fo.types.YOLOv5Dataset,
    classes=classes_of_interest,
    export_media=True,  # Copy the images to the export directory
)

print(f"Training dataset exported to: {train_export_dir}")
print(f"Validation dataset exported to: {val_export_dir}")
