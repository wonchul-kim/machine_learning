import os
import os.path as osp 
import glob 
from shutil import copyfile
from tqdm import tqdm
import random

def split_dataset_v1(input_dir, output_dir, img_exts):
    if not osp.exists(output_dir):
        os.mkdir(output_dir)

    folders1 = [folder.split("/")[-1] for folder in glob.glob(osp.join(input_dir, "**")) if not osp.isfile(folder)]
    for folder1 in folders1:

        folders2 = [folder.split("/")[-1] for folder in glob.glob(osp.join(input_dir, folder1, "**")) if not osp.isfile(folder)]
        for folder2 in tqdm(folders2, desc=folder1):
            
            json_files = glob.glob(osp.join(input_dir, folder1, folder2, '*.json'))
            
            for json_file in json_files:
                
                if random.random() < 0.1:
                    _output_dir = osp.join(output_dir, 'val')
                else:
                    _output_dir = osp.join(output_dir, 'train')

                if not osp.exists(_output_dir):
                    os.mkdir(_output_dir)
                    
                filename = osp.split(osp.splitext(json_file)[0])[-1]
                moved = False
                for img_ext in img_exts:
                    img_file = osp.splitext(json_file)[0] + f'.{img_ext}'
                    
                    if osp.exists(img_file):
                        copyfile(img_file, osp.join(_output_dir, filename + f'.{img_ext}'))
                        copyfile(json_file, osp.join(_output_dir, filename + '.json'))
                        moved = True 
                assert moved, RuntimeError(f"There is no such image file: {osp.join(input_dir, filename)} with {img_exts}")


def split_dataset_v2(input_dir, output_dir, img_exts):
    if not osp.exists(output_dir):
        os.mkdir(output_dir)

    folders = [folder.split("/")[-1] for folder in glob.glob(osp.join(input_dir, "**")) if not osp.isfile(folder)]
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
            moved = False
            for img_ext in img_exts:
                img_file = osp.splitext(json_file)[0] + f'.{img_ext}'
                
                if osp.exists(img_file):
                    copyfile(img_file, osp.join(_output_dir, filename + f'.{img_ext}'))
                    copyfile(json_file, osp.join(_output_dir, filename + '.json'))
                    moved = True 
            assert moved, RuntimeError(f"There is no such image file: {osp.join(input_dir, filename)} with {img_exts}")

                
if __name__ == '__main__':
    input_dir = '/DeepLearning/etc/_athena_tests/benchmark/interojo/rect/raw_dataset'
    output_dir = '/DeepLearning/etc/_athena_tests/benchmark/interojo/rect/split_dataset'
    
    img_exts = ['bmp']
    split_dataset_v2(input_dir, output_dir, img_exts)


            
        
        
        
    