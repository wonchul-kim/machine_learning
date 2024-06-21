import dota_utils as util
import os
import os.path as osp
import numpy as np
from PIL import Image


## trans dota format to format YOLO(darknet) required
def dota2darknet(imgpath, txtpath, dstpath, extractclassname):
    """
    :param imgpath: the path of images
    :param txtpath: the path of txt in dota format
    :param dstpath: the path of txt in YOLO format
    :param extractclassname: the category you selected
    :return:
    """
    filelist = util.GetFileFromThisRootDir(txtpath)
    for fullname in filelist:
        objects = util.parse_dota_poly(fullname)
        name = os.path.splitext(os.path.basename(fullname))[0]
        img_fullname = os.path.join(imgpath, name + '.bmp')
        img = Image.open(img_fullname)
        img_w, img_h = img.size
        # print img_w,img_h
        
        if not osp.exists(dstpath):
            os.mkdir(dstpath)
        
        with open(os.path.join(dstpath, name + '.txt'), 'w') as f_out:
            for obj in objects:
                poly = obj['poly']
                bbox = np.array(util.dots4ToRecC(poly, img_w, img_h))
                if (sum(bbox <= 0) + sum(bbox >= 1)) >= 1:
                    continue
                if (obj['name'] in extractclassname):
                    id = extractclassname.index(obj['name'])
                else:
                    continue
                outline = str(id) + ' ' + ' '.join(list(map(str, bbox)))
                f_out.write(outline + '\n')

if __name__ == '__main__':
    ## an example
    dota2darknet('/HDD/datasets/projects/rich/24.06.12/split_dataset_dota/train/images',
                 '/HDD/datasets/projects/rich/24.06.12/split_dataset_dota/train/labelTxt',
                 '/HDD/datasets/projects/rich/24.06.12/split_dataset_dota/train/labels',
                 ['BOX', 'EGG'])