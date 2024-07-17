import os
import os.path as osp 
import glob 
import json 
import numpy as np
from tqdm import tqdm 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
sns.set_style('darkgrid')
from mlearning.utils.charts import draw_pie_chart, draw_dist_chart
from pdf import create_pdf

def analyze_labelme(input_dir, output_dir, project_name, title, subtitle):
    if not osp.exists(output_dir):
        os.mkdir(output_dir)
        
    modes = [folder.split("/")[-1] for folder in glob.glob(osp.join(input_dir, "**")) if not osp.isfile(folder)]

    num_images = {}
    labels_by_mode = {}
    labels_by_image = {}
    objects_by_label = {}

    for mode in modes:
        json_files = glob.glob(osp.join(input_dir, mode, '*.json'))
        
        mode_labels = {}
        count = 0
        for json_file in tqdm(json_files, desc=mode):
            count += 1
            image_labels = {}
            filename = osp.split(osp.splitext(json_file)[0])[-1]
            with open(json_file, 'r') as jf:
                anns = json.load(jf)['shapes']
            
            xs, ys = [], []
            for ann in anns:
                label = ann['label']
                # labels by mode
                if label in mode_labels.keys():
                    mode_labels[label] += 1
                else:
                    mode_labels[label] = 1

                # labels by image
                if label in image_labels.keys():
                    image_labels[label] += 1
                else:
                    image_labels[label] = 1
                    
                # objects
                points = ann['points']
                for point in points:
                    xs.append(point[0])
                    ys.append(point[1])
                width = np.max(xs) - np.min(xs)
                height = np.max(ys) - np.min(ys)
                if label in objects_by_label.keys():
                    objects_by_label[label]['width'].append(width)
                    objects_by_label[label]['height'].append(height)
                else:
                    objects_by_label[label] = {'width': [width], 'height': [height]}
                    
            image_labels.update({"mode": mode})
            labels_by_image.update({filename: image_labels})
            
        num_images.update({mode: count})
        labels_by_mode.update({mode: mode_labels})
        
    # number of images
    plt.figure(figsize=(8, 6))
    bars = plt.bar(list(num_images.keys()), list(num_images.values()), color=['blue', 'green'])
    for bar, value in zip(bars, list(num_images.values())):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, str(value), ha='center', va='bottom')
    plt.title('Number of Images')
    plt.xlabel('Dataset')
    plt.ylabel('Number of Images')
    plt.grid(True)
    plt.savefig(osp.join(output_dir, 'num_images_bar.png'))
    draw_pie_chart(num_images, 'Number of Images', osp.join(output_dir, f'num_images_pie.png'))


    # labels by mode    
    for key, val in labels_by_mode.items():
        draw_pie_chart(val, key, osp.join(output_dir, f'{key}_pie_chart.png'))
        
    # labels by image
    df_labels_by_image = pd.DataFrame(labels_by_image).T
    df_labels_by_image.name = 'filename'
    df_labels_by_image.fillna(0, inplace=True)
    df_labels_by_image.to_csv(osp.join(output_dir, 'labels_by_image.csv'))
    html_labels_by_image = df_labels_by_image.to_html()
    with open(osp.join(output_dir, 'labels_by_image.html'), 'w') as f:
        f.write(html_labels_by_image)

    plt.figure(figsize=(6, 6))  
    for key, val in objects_by_label.items():
        plt.scatter(val['width'], val['height'], marker='o', s=5, label=key)
    plt.title('Object Sizes')  
    plt.xlabel('Width') 
    plt.ylabel('Height')
    plt.grid(True) 
    plt.legend()
    plt.savefig(osp.join(output_dir, 'objects_size_by_labels.png'))

    for key, val in objects_by_label.items():
        draw_dist_chart(val, key, osp.join(output_dir, f'objects_size_{key}.png'))



    create_pdf(title, subtitle, output_dir)

if __name__ == '__main__':
    # input_dir = '/DeepLearning/etc/_athena_tests/benchmark/sungwoo/inner_body/split_dataset'
    # output_dir = '/DeepLearning/etc/_athena_tests/benchmark/sungwoo/inner_body/reports'

    # project_name = 'Sungwoo-inner-body'
    # title = "Dataset Analysis"
    # subtitle = f"The dataset is {project_name}"
    
    # analyze_labelme(input_dir, output_dir, project_name, title, subtitle)
    
    input_dir = '/DeepLearning/etc/_athena_tests/benchmark/interojo/rect/split_dataset'
    output_dir = '/DeepLearning/etc/_athena_tests/benchmark/interojo/reports'

    project_name = 'Rich-Box'
    title = "Dataset Analysis"
    subtitle = f"The dataset is {project_name}"
    
    analyze_labelme(input_dir, output_dir, project_name, title, subtitle)
    
    