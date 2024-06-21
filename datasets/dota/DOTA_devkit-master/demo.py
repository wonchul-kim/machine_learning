import numpy as np
import matplotlib.pyplot as plt
import os
from DOTA import DOTA
import dota_utils as util
import pylab
pylab.rcParams['figure.figsize'] = (20.0, 20.0)

# example = DOTA('/HDD/_projects/github/machine_learning/DOTA_devkit-master/example')
# imgids = example.getImgIds(catNms=['ship', 'storage-tank'])
example = DOTA('/HDD/datasets/projects/rich/split_dataset_dota/val')
imgids = example.getImgIds(catNms=['BOX'])
imgid = imgids[1]
img = example.loadImgs(imgid)[0]

plt.axis('off')

plt.imshow(img)
plt.show()
plt.savefig('dot.png')


anns = example.loadAnns(imgId=imgid)
# print(anns)
example.showAnns(anns, imgid, 2)