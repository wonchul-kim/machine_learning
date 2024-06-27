import glob 
import os.path as osp
from ultralytics import YOLO

# Load a model
# model = YOLO("yolov8n-obb.pt")  # load an official model
model = YOLO("/HDD/_projects/github/machine_learning/runs/obb/train8/weights/best.pt")  # load a custom model

# Predict with the model

input_dir = '/Data/01.Image/리치코리아/LA/24.06.21/test/data'
img_files = glob.glob(osp.join(input_dir, '*.bmp'))

for img_file in img_files:
    results = model(img_file, save=True, imgsz=1024, conf=0.2)  # predict on an image