from typing import Union, Dict
from ultralytics import YOLO, settings
settings.update({'wandb': False})

from mlearning.algs.roboflow.utils.parsing import get_cfg, get_params, get_weights

def train(task: str, model_name: str, backbone: str, data: str, cfg: Union[str, Dict], params: Union[str, Dict]):

    cfg = get_cfg(cfg)
    params = get_params(params)
    weights = get_weights(task, model_name, backbone)
    
    model = YOLO(weights)
    print(f"* Successfully LOADED model")
    
    print(f">>> Start to train")
    model.train(data=data, **cfg, **params)
    print(f"* FINISHED training --------------------------------------")
    
if __name__ == '__main__':
    task = 'obb'
    model_name = 'yolov8'
    backbone = 'n'
    data = '/HDD/datasets/projects/mlearning/yolo/obb_detection/split_dataset_yolo_obb/data.yaml'
    cfg = '/HDD/datasets/projects/mlearning/yolo/obb_detection/split_dataset_yolo_obb/cfg.yaml'
    params = '/HDD/datasets/projects/mlearning/yolo/obb_detection/split_dataset_yolo_obb/train.yaml'
    
    train(task, model_name, backbone, data, cfg, params)