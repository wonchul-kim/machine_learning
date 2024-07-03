import glob 
import os
import os.path as osp 
import json 
from tqdm import tqdm


def labelme2yolo_instance_segmentation(input_dir, output_dir, image_ext,
                                       copy_image=True, image_width=None, image_height=None):

    if not osp.exists(output_dir):
        os.mkdir(output_dir)
        
    folders = [folder.split("/")[-1] for folder in glob.glob(osp.join(input_dir, "**")) if not osp.isfile(folder)]


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
                anns = json.load(jf)['shapes']
                
            if copy_image:
                import cv2
                from shutil import copyfile
                img_file = osp.join(input_dir, folder, filename + f'.{image_ext}')
                assert osp.exists(img_file), ValueError(f"There is no such image: {img_file}")
                
                image = cv2.imread(img_file)
                image_width = image.shape[1]
                image_height = image.shape[0]
                _output_image_dir = osp.join(output_dir, 'images', folder)
                if not osp.exists(_output_image_dir):
                    os.makedirs(_output_image_dir)
                copyfile(img_file, osp.join(_output_image_dir, filename + f'.{image_ext}'))
                
                
            assert image_width != None and image_height != None, ValueError(f"The input size (width, height) must be assigned")
                
            if len(anns) != 0:
                for ann in anns:
                    shape_type = ann['shape_type']
                    label = ann['label']
                    if label not in class2idx.keys():
                        class2idx.update({label: len(class2idx)})
                    points = ann['points']
                    if shape_type == 'point' or len(points) <= 2:
                        continue
                    assert len(points) >= 3, RuntimeError(f"The number of polygon points must be more than 3, not {len(points)} with {shape_type}")

                    txt.write(str(class2idx[label]) + ' ')
                    for idx, point in enumerate(points):
                        if idx == len(points) -1:
                            txt.write(f'{round(point[0]/image_width, 3)} {round(point[1]/image_height, 3)}')
                        else:
                            txt.write(f'{round(point[0]/image_width, 3)} {round(point[1]/image_height, 3)} ')
                    txt.write('\n')
            
        txt.close()

    txt = open(osp.join(output_dir, 'classes.txt'), 'w')
    for key, val in class2idx.items():
        txt.write(f'{val}: {key}\n')
    txt.close()
            
            
            
if __name__ == '__main__':
    input_dir = '/DeepLearning/etc/_athena_tests/benchmark/sungwoo/inner_body/split_dataset'
    output_dir = '/DeepLearning/etc/_athena_tests/benchmark/sungwoo/inner_body/split_dataset_yolo_is'

    copy_image = True
    image_ext = 'bmp'
    
    image_width = None
    image_height = None
    
    labelme2yolo_instance_segmentation(input_dir, output_dir, image_ext, copy_image, image_width, image_height)