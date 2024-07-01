from ultralytics import YOLO 

model = YOLO("yolov8n-seg.pt")

results = model.train(data='interojo-seg.yaml', epochs=100, imgsz=640, device='0,1')