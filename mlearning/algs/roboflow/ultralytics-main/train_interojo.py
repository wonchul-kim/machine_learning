from ultralytics import YOLO 

model = YOLO("yolov8m.pt")

results = model.train(data='interojo-od.yaml', epochs=300, imgsz=832, device='0,1', batch=8)