from ultralytics import YOLO 

model = YOLO("yolov8n-seg.pt")

results = model.train(data='interojo-seg.yaml', epochs=100, imgsz=60, device=0)