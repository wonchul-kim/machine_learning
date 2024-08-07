from ultralytics import YOLO 

model = YOLO("/HDD/weights/yolov10/obb/yolov10n-obb.pt")

results = model.train(data='ktg_obb.yaml', epochs=300, imgsz=1024, device='0,1', batch=2)