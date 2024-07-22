import glob 
import os
import os.path as osp 
import json 
import numpy as np
from utils import xyxy2xywh
from tqdm import tqdm



def labelme2yolo_object_detection(input_dir, output_dir, copy_image, image_exts):

    if not osp.exists(output_dir):
        os.mkdir(output_dir)
        
    folders = [folder.split("/")[-1] for folder in glob.glob(osp.join(input_dir, "**")) if not osp.isfile(folder)]

    image_width, image_height = None, None

    class2idx = {}
    for folder in folders:
        _output_labels_dir = osp.join(output_dir, 'labels', folder)
        if not osp.exists(_output_labels_dir):
            os.makedirs(_output_labels_dir)

        json_files = glob.glob(osp.join(input_dir, folder, '*.json'))
        print(f"There are {len(json_files)} json files")

        for json_file in tqdm(json_files, desc=folder):
            filename = osp.split(osp.splitext(json_file)[0])[-1]
            txt = open(osp.join(_output_labels_dir, filename + '.txt'), 'w')
            with open(json_file, 'r') as jf:
                _anns = json.load(jf)
                anns = _anns['shapes']
                
                if 'imageHeight' in _anns and 'imageWidth' in _anns:
                    image_height, image_width = _anns['imageHeight'], _anns['imageWidth']
                
            if copy_image:
                import cv2
                from shutil import copyfile
                img_file = osp.join(input_dir, folder, filename + f'.{image_ext}')
                image = cv2.imread(img_file)
                image_width = image.shape[1]
                image_height = image.shape[0]
                _output_image_dir = osp.join(output_dir, 'images', folder)
                if not osp.exists(_output_image_dir):
                    os.makedirs(_output_image_dir)
                copyfile(img_file, osp.join(_output_image_dir, filename + f'.{image_ext}'))
                
            if len(anns) != 0:
                for ann in anns:
                    shape_type = ann['shape_type']
                    label = ann['label']
                    if label not in class2idx.keys():
                        class2idx.update({label: len(class2idx)})
                    points = ann['points']
                    xyxy = []
                    if shape_type == 'rectangle':
                        for point in points:
                            xyxy.append(point[0])
                            xyxy.append(point[1])

                        if len(xyxy) > 4:
                            raise RuntimeError(f"shape type is rectangle, but there are more than 4 points")

                    elif shape_type == 'polygon' or shape_type == 'Watershed':
                        xs, ys = [], []
                        for point in points:
                            xs.append(point[0])
                            ys.append(point[1])

                        xyxy.append(np.max(xs))
                        xyxy.append(np.max(ys))
                        xyxy.append(np.min(xs))
                        xyxy.append(np.min(ys))
                    else: 
                        print(f"NotImplemented shape: {shape_type} for {json_file}")
                        continue

                        
                    assert image_width is not None and image_height is not None, RuntimeError(f"Image width is {image_width} and image height is {image_height}")
                    xywh = xyxy2xywh([image_height, image_width], xyxy)
                    txt.write(str(class2idx[label]) + ' ')
                    for kdx in range(len(xywh)):
                        if kdx == len(xywh) -1:
                            txt.write(str(round(xywh[kdx], 3)))
                        else:
                            txt.write(str(round(xywh[kdx], 3)) + ' ')
                    txt.write('\n')
            
        txt.close()

    txt = open(osp.join(output_dir, 'classes.txt'), 'w')
    for key, val in class2idx.items():
        txt.write(f'{key}: {val}\n')
    txt.close()
            
            
if __name__ == '__main__':
    input_dir = '/storage/projects/ktg/24.07.10/split_dataset'
    output_dir = '/storage/projects/ktg/24.07.10/split_datasets_yolo_hbb'

    copy_image = True
    image_ext = 'bmp'
    
    labelme2yolo_object_detection(input_dir, output_dir, copy_image, image_ext)