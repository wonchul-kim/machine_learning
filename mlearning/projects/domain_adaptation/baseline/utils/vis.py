import os
import os.path as osp
import cv2 
import imgviz
import numpy as np
import torch
import random

def vis_seg_dataset(dataset, output_dir, ratio=0.1):
    
    if not osp.exists(output_dir):
        os.mkdir(output_dir)
    color_map = imgviz.label_colormap(50)

    for batch in dataset:
        if ratio < random.random():
            continue
        image, mask, filename = batch
        if isinstance(image, torch.Tensor):
            image = image.numpy().transpose(1, 2, 0)
            mask = mask.numpy()
        
        mask = color_map[mask].astype(np.uint8)
        vis_image = cv2.addWeighted(image, 0.8, mask, 0.2, 0)
        cv2.imwrite(osp.join(output_dir, filename + '.png'), vis_image)                                                                                          