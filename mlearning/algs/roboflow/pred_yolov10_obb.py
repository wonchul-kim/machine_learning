from ultralytics import YOLO 

model = YOLO("/HDD/weights/yolov10/obb/yolov10n-obb.pt")

model.predict('/path/to/test.jpg', save=True, imgsz=1024, conf=0.5)
