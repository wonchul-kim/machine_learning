from ultralytics import YOLO 

model = YOLO("yolov8l-seg.pt")

results = model.train(data='sungwoo-u-gap.yaml', epochs=300, imgsz=768, device='0,1,2,3', batch=16)
