import os
import os.path as osp

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

from dataset import CustomDataset
from models.unet import UNet
from PIL import Image
from utils.functional import rle_encode
import numpy as np
import pandas as pd

output_dir = '/HDD/datasets/public/dacon_domain_adaptation/outputs'
vis_dataset = False

if not osp.exists(output_dir):
    os.mkdir(output_dir)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = A.Compose(
    [   
        A.Resize(224, 224),
        A.Normalize(),
        ToTensorV2()
    ]
)

test_dataset = CustomDataset(root_dir='/HDD/datasets/public/dacon_domain_adaptation',
                             csv_file='./test.csv', transform=transform, infer=True)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)

model = UNet().to(device)

with torch.no_grad():
    model.eval()
    result = []
    for images in tqdm(test_dataloader):
        images = images.float().to(device)
        outputs = model(images)
        outputs = torch.softmax(outputs, dim=1).cpu()
        outputs = torch.argmax(outputs, dim=1).numpy()
        # batch에 존재하는 각 이미지에 대해서 반복
        for pred in outputs:
            pred = pred.astype(np.uint8)
            pred = Image.fromarray(pred) # 이미지로 변환
            pred = pred.resize((960, 540), Image.NEAREST) # 960 x 540 사이즈로 변환
            pred = np.array(pred) # 다시 수치로 변환
            # class 0 ~ 11에 해당하는 경우에 마스크 형성 / 12(배경)는 제외하고 진행
            for class_id in range(12):
                class_mask = (pred == class_id).astype(np.uint8)
                if np.sum(class_mask) > 0: # 마스크가 존재하는 경우 encode
                    mask_rle = rle_encode(class_mask)
                    result.append(mask_rle)
                else: # 마스크가 존재하지 않는 경우 -1
                    result.append(-1)

submit = pd.read_csv('./sample_submission.csv')
submit['mask_rle'] = result
submit.to_csv('./baseline_submit.csv', index=False)