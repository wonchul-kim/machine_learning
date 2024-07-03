from ultralytics import YOLO 

model = YOLO("yolov8m-seg.pt")

results = model.train(data='sungwoo-inner-body.yaml', epochs=300, imgsz=1024, device='0,1', batch=4)