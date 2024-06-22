from ultralytics import YOLO

# Load a model
# model = YOLO("yolov8n-obb.pt")  # load an official model
model = YOLO("/HDD/_projects/github/machine_learning/runs/obb/train8/weights/best.pt")  # load a custom model

# Predict with the model
results = model("/HDD/datasets/projects/rich/24.06.12/split_dataset_yolo/images/val/21_124060517242360_1_rgb.bmp",
                save=True, imgsz=1024, conf=0.2)  # predict on an image
