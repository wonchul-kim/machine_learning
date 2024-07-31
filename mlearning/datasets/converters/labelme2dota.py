import glob 
import os
import os.path as osp 
from shutil import copyfile
import json 
from tqdm import tqdm
from mlearning.datasets.converters.polygon2dota import polygon2dota_by_rotate

def convert_labelme2dota(input_dir, output_dir, copy_image=True, image_ext='bmp'):
    if not osp.exists(output_dir):
        os.mkdir(output_dir)
        
    folders = [folder.split("/")[-1] for folder in glob.glob(osp.join(input_dir, "**")) if not osp.isfile(folder)]

    for folder in folders:
        _output_label_dir = osp.join(output_dir, folder, 'labelTxt')
        if not osp.exists(_output_label_dir):
            os.makedirs(_output_label_dir)

        if copy_image:
            _output_image_dir = osp.join(output_dir, folder, 'images')
            if not osp.exists(_output_image_dir):
                os.makedirs(_output_image_dir)

        json_files = glob.glob(osp.join(input_dir, folder, '*.json'))
        print(f"There are {len(json_files)} json files")

        for json_file in tqdm(json_files, desc=folder):
            
            assert osp.exists(json_file), ValueError(f"There is no such json-file: {json_file}")
            
            filename = osp.split(osp.splitext(json_file)[0])[-1]
            
            if copy_image:
                img_file = osp.splitext(json_file)[0] + f'.{image_ext}'
                assert osp.exists(json_file), ValueError(f"There is no such json-file: {json_file}")

                copyfile(img_file, osp.join(_output_image_dir, filename + f'.{image_ext}'))

            
            txt = open(osp.join(_output_label_dir, filename + '.txt'), 'w')
            with open(json_file, 'r') as jf:
                anns = json.load(jf)['shapes']
                
            if len(anns) != 0:
                for ann in anns:
                    points, _ = polygon2dota_by_rotate(ann['points'])
                    data = ''
                    for point in points:
                        data += f'{str(int(point[0]))} {str(int(point[1]))} '
                    
                    data += f'{ann["label"]} 0'
                    txt.write(data)
                    txt.write("\n")
            
        txt.close()

            
