from ultralytics import YOLO 

model = YOLO("yolov8n.pt")

results = model.train(data='interojo.yaml', epochs=1000, imgsz=180)