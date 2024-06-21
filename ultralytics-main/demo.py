from ultralytics import YOLO 

model = YOLO("yolov8n-obb.yaml")

results = model.train(data='rich.yaml', epochs=100, imgsz=128)