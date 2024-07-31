from mlearning.datasets.converters.labelme2dota import convert_labelme2dota
from mlearning.datasets.converters.dota2yolo import convert_dota2yolo_obb

input_dir = '/HDD/datasets/projects/kt_g/datasets/split_dataset'
output_dir = '/HDD/datasets/projects/kt_g/datasets/split_dataset_dota'

convert_labelme2dota(input_dir, output_dir, copy_image=True, image_ext='bmp')

input_dir = output_dir
output_dir = '/HDD/datasets/projects/kt_g/datasets/split_dataset_yolo_obb'

convert_dota2yolo_obb(input_dir, output_dir, copy_image=True, image_ext='bmp')