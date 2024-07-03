import os 
import os.path as osp
import glob
import random
from shutil import copyfile
from tqdm import tqdm

def split_dataset(input_dir, output_dir):
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
            img_file = osp.splitext(json_file)[0] + '.jpg'
            assert osp.exists(img_file), RuntimeError(f"There is no such image file: {img_file}")

            copyfile(img_file, osp.join(_output_dir, filename + '.jpg'))
            copyfile(json_file, osp.join(_output_dir, filename + '.json'))
                
if __name__ == '__main__':
    input_dir = '/HDD/datasets/public/duts/raw_dataset/DUTS-TR'
    output_dir = '/HDD/datasets/public/duts/split_dataset'
    
    split_dataset(input_dir, output_dir)
