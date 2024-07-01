from ultralytics import YOLO 

model = YOLO("yolov8m-seg.pt")

results = model.train(data='sungwoo-inner-body.yaml', epochs=300, imgsz=2048, device='0,1,2,3', batch=4)