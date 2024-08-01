import glob
import json
import os
import os.path as osp
import numpy as np
import cv2

input_dir = "/HDD/datasets/projects/sungjin/body/test/patches"
output_dir = "/HDD/datasets/projects/sungjin/body/test/vis"

if not osp.exists(output_dir):
    os.mkdir(output_dir)

# TODO: need to consider all shape-types

img_files = glob.glob(osp.join(input_dir, '*.bmp'))

for img_file in img_files:
    filename = osp.split(osp.splitext(img_file)[0])[-1]
    json_file = osp.splitext(img_file)[0] + '.json'

    if osp.exists(json_file):
        img = cv2.imread(img_file)
        height, width, _ = img.shape

        with open(json_file) as jf:
            anns = json.load(jf)

        for ann in anns['shapes']:
            label = ann['label']
            shape_type = ann['shape_type']
            points = ann['points']

            if shape_type == 'rectangle':
                cv2.rectangle(img, (int(points[0][0]), int(points[0][1])), (int(points[1][0]), int(points[1][1])),
                            (0, 0, 255), 5)
            elif shape_type == 'polygon':
                cv2.fillPoly(img, [np.array(points, dtype=np.int32)], color=(255, 255, 0))
            else:
                raise NotImplementedError(f"There is no such shape-type({shape_type}) considered")


        # cv2.line(img, (0, 256), (width, 256), (0, 0, 255), 2) #horizontal
        # cv2.line(img, (408, 0), (408, height), (0, 0, 255), 2) #vertical
        # cv2.line(img, (664, 0), (664, height), (0, 0, 255), 2) #vertical
        # cv2.line(img, (512, 0), (512, height), (0, 0, 255), 2)

        cv2.imwrite(osp.join(output_dir, filename + '.bmp'), img)
