import os
import os.path as osp 
import yaml 
from typing import Union, Dict
from pathlib import Path 
import warnings

def get_cfg(cfg: Union[str, Dict]):
    
    if isinstance(cfg, str):
        assert osp.exists(cfg), ValueError(f"There is no such cfg at {cfg}")
        with open(cfg) as yf:
            cfg = yaml.load(yf)
    assert isinstance(cfg, dict), ValueError(f"Parameters must be dict, not {type(cfg)} which is {cfg}")    
    print(f"* Successfully LOADED cfg: {cfg}")
    
    if 'output_dir' in cfg:
        if not osp.exists(cfg['output_dir']):
            os.mkdir(cfg['output_dir'])
        cfg['project'] = Path(cfg['output_dir'])
        del cfg['output_dir']
    else:
        warnings.warn(f"There is no output-dir assigned")
    return cfg


def get_params(params: Union[str, Dict]):
    if isinstance(params, str):
        assert osp.exists(params), ValueError(f"There is no such params at {params}")
        with open(params) as yf:
            params = yaml.load(yf)
    assert isinstance(params, dict), ValueError(f"Parameters must be dict, not {type(params)} which is {params}")    
    print(f"* Successfully LOADED params: {params}")

    return params


def get_weights(task: str, model_name: str, backbone: str):
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
    print(f"* Successfully DEFINED weights: {weights}")

    return weights