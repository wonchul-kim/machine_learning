import yaml
import os.path as osp
from ultralytics import YOLO 

def train(task, model_name, backbone, data, cfg):

    if isinstance(cfg, str):
        assert osp.exists(cfg), ValueError(f"There is no such cfg at {cfg}")

        with open(cfg) as yf:
            params = yaml.load(yf)

    weights = None
    if task == 'detection' or task == 'det':
        weights = f'{model_name}{backbone}.pt'
    elif 'obb' in task:
        weights = f'{model_name}{backbone}-obb.pt'
    elif 'seg' in task:
        weights = f'{model_name}{backbone}-seg.pt'
    else:
        NotImplementedError(f"There is no such weights for {model_name} and {backbone}")
    
    assert weights is not None, RuntimeError(f"weights is None")
    
    model = YOLO(weights)

    params = {'epochs': 300, 'imgsz': 1024, 'device': '0,1', 'batch': 2}
    assert isinstance(params, dict), ValueError(f"Parameters must be dict, not {type(params)} which is {params}")    
    results = model.train(data=data, **params)
    
if __name__ == '__main__':
    task = 'det'
    model_name = 'yolov8'
    backbone = 'l'
    data='ktg_obb.yaml'
    cfg = {}
    
    train(task, model_name, backbone, data, cfg)