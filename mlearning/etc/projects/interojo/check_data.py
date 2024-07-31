import os.path as osp 
import glob 
import json
from tqdm import tqdm


                
def check_data_v1(input_dir):
    
    num_data = 0
    shape_types = set()
    classes = set()

    folders1 = [folder.split("/")[-1] for folder in glob.glob(osp.join(input_dir, "**")) if not osp.isfile(folder)]
    for folder1 in folders1:

        folders2 = [folder.split("/")[-1] for folder in glob.glob(osp.join(input_dir, folder1, "**")) if not osp.isfile(folder)]
        for folder2 in tqdm(folders2, desc=folder1):
            
            json_files = glob.glob(osp.join(input_dir, folder1, folder2, '*.json'))
            
            for json_file in json_files:
                with open(json_file, 'r') as jf:
                    anns = json.load(jf)['shapes']
                    
                for ann in anns:
                    shape_type = ann['shape_type']
                    shape_types.add(shape_type)
                    classes.add(ann['label'])
            num_data += 1

    print(shape_types)                
    print(classes)
    print(num_data)
                
def check_data_v2(input_dir):

    num_data = 0
    shape_types = set()
    classes = set()

    folders = [folder.split("/")[-1] for folder in glob.glob(osp.join(input_dir, "**")) if not osp.isfile(folder)]
    for folder in tqdm(folders):
        
        json_files = glob.glob(osp.join(input_dir, folder, '*.json'))
        
        for json_file in json_files:
            with open(json_file, 'r') as jf:
                anns = json.load(jf)['shapes']
                
            for ann in anns:
                shape_type = ann['shape_type']
                shape_types.add(shape_type)
                classes.add(ann['label'])

            num_data += 1

    print(shape_types)                
    print(classes)
    print(num_data)
                
                
                
if __name__ == '__main__':
    input_dir = '/Data/01.Image/interojo/3rd_poc/211209_분리_ver2'
    
    check_data_v2(input_dir)
