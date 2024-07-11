import numpy as np
from scipy.ndimage.morphology import distance_transform_edt
import library.data.cityscapes_labels as cityscapes_labels
from PIL import Image 
import torch 


def mask_to_onehot(mask, num_classes):
    _mask = [mask == (i + 1) for i in range(num_classes)]
    return np.array(_mask).astype(np.uint8)


def onehot_to_binary_edges(mask, radius, num_classes):        
    mask_pad = np.pad(mask, ((0, 0), (1, 1), (1, 1)), mode='constant', constant_values=0)
    edgemap = np.zeros(mask.shape[1:])
    for i in range(num_classes):
        dist = distance_transform_edt(mask_pad[i, :])+distance_transform_edt(1.0-mask_pad[i, :])
        dist = dist[1:-1, 1:-1]
        dist[dist > radius] = 0
        edgemap += dist
    edgemap = np.expand_dims(edgemap, axis=0)    
    edgemap = (edgemap > 0).astype(np.uint8)
    return edgemap

id_to_trainid = cityscapes_labels.label2trainid
num_classes = 19

mask = Image.open('/HDD/datasets/public/cityscapes/gtFine/val/frankfurt/frankfurt_000000_000294_gtFine_labelIds.png').convert('RGB')

mask = np.array(mask)

mask = mask.copy()
for k, v in id_to_trainid.items():
    binary_mask = (mask == k) 
    mask[binary_mask] = v

mask = Image.fromarray(mask.astype(np.uint8))

_edgemap = mask.copy()
_edgemap = mask_to_onehot(_edgemap, num_classes)
_edgemap = onehot_to_binary_edges(_edgemap, 2, num_classes)
edgemap = torch.from_numpy(_edgemap).float()