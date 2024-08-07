from ultralytics import YOLO


def export(weights, target):
    model = YOLO(weights)
    
    if target == 'onnx':
        model.export(format='onnx', opset=14)
    else:
        NotImplementedError(f'There is no such case for {target}')

# model = YOLO('/DeepLearning/_projects/kt_g/outputs/yolov8_obb/train4/weights/yolov8_obb_3328.onnx')

# preds = model('/DeepLearning/_projects/kt_g/split_dataset/val/3_124071716123953_1_image_1.bmp')
