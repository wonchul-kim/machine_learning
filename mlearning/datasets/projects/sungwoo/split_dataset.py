import os
import os.path as osp 
import glob 
from shutil import copyfile
from tqdm import tqdm
import random

def split_dataset(input_dir, output_dir, image_ext):
    if not osp.exists(output_dir):
        os.makedirs(output_dir)

    folders = [folder.split("/")[-1] for folder in glob.glob(osp.join(input_dir, "**")) if not osp.isfile(folder)]
    
    if len(folders) == 0:
        folders = ['./']
    for folder in folders:
        
        json_files = glob.glob(osp.join(input_dir, folder, '*.json'))
        
        for json_file in tqdm(json_files, desc=folder):
            
            if random.random() < 0.1:
                _output_dir = osp.join(output_dir, 'val')
            else:
                _output_dir = osp.join(output_dir, 'train')

            if not osp.exists(_output_dir):
                os.mkdir(_output_dir)
                
            filename = osp.split(osp.splitext(json_file)[0])[-1]
            img_file = osp.splitext(json_file)[0] + f'.{image_ext}'
            assert osp.exists(img_file), RuntimeError(f"There is no such image file: {img_file}")

            copyfile(img_file, osp.join(_output_dir, filename + f'.{image_ext}'))
            copyfile(json_file, osp.join(_output_dir, filename + '.json'))
                
if __name__ == '__main__':
    # input_dir = '/Data/01.Image/yb/24.07.01/u_gap'
    # output_dir = '/DeepLearning/etc/_athena_tests/benchmark/sungwoo/u_gap/split_dataset'
    # image_ext = 'bmp'
    
    # split_dataset(input_dir, output_dir, image_ext)

    input_dir = '/Data/01.Image/yb/24.07.01/inner'
    output_dir = '/DeepLearning/etc/_athena_tests/benchmark/sungwoo/inner_body/split_dataset'
    image_ext = 'bmp'
    
    split_dataset(input_dir, output_dir, image_ext)
            
        
        
        
    