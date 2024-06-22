from ultralytics import YOLO 

model = YOLO("yolov8m-obb.yaml")

results = model.train(data='rich.yaml', epochs=1000, imgsz=1024)