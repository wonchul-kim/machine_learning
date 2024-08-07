from ultralytics import YOLO 

model = YOLO("yolov8m.pt")

results = model.train(data='/HDD/weights/yolov10/interojo-od.yaml', epochs=300, imgsz=832, device='0,1', batch=8)