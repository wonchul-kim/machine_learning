import os
import os.path as osp

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

from dataset import CustomDataset
from models.unet import UNet
from utils.vis import vis_seg_dataset

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
if vis_dataset:
    dataset = CustomDataset(root_dir='/HDD/datasets/public/dacon_domain_adaptation', 
                            csv_file='train_source.csv', transform=None)

    vis_seg_dataset(dataset, osp.join(output_dir, 'vis'))

dataset = CustomDataset(root_dir='/HDD/datasets/public/dacon_domain_adaptation', 
                        csv_file='train_source.csv', transform=transform)

dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)

model = UNet().to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(20):  # 20 에폭 동안 학습합니다.
    model.train()
    epoch_loss = 0
    for images, masks, filename in tqdm(dataloader):
        images = images.float().to(device)
        masks = masks.long().to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks.squeeze(1))
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f'Epoch {epoch+1}, Loss: {epoch_loss/len(dataloader)}')