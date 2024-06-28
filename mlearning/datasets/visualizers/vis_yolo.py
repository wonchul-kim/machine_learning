import cv2 
import glob 
import os
import os.path as osp 
# from ..converters.utils import xywh2xyxy
def xywh2xyxy(imgsz, xywh):

    if not len(imgsz) >= 2:
        raise RuntimeError(f"imgsz should be [height, width, channel] or [height, width]")
    elif imgsz[0] <= 3:
        raise RuntimeError(f"imgsz should be [height, width, channel] or [height, width]")

    if isinstance(xywh, list):
        x, y, w, h = xywh[0], xywh[1], xywh[2], xywh[3]
    else:
        raise RuntimeError(f"xywh must be list such as [x, y, w, h]")

    dh, dw = imgsz[0], imgsz[1]
    l = ((x - w / 2) * dw) # x0
    r = ((x + w / 2) * dw) # x1
    t = ((y - h / 2) * dh) # y0
    b = ((y + h / 2) * dh) # y1
    
    if l < 0:
        l = 0
    if r > dw - 1:
        r = dw - 1
    if t < 0:
        t = 0
    if b > dh - 1:
        b = dh - 1   

    return [l, t, r, b] # x0, y0, x1, y1


input_dir = '/HDD/datasets/projects/interojo/split_datasets_yolo'
output_dir = '/HDD/datasets/projects/interojo/vis'
image_ext = 'png'

if not osp.exists(output_dir):
    os.mkdir(output_dir)


folders = [folder.split("/")[-1] for folder in glob.glob(osp.join(input_dir, "images/**")) if not osp.isfile(folder)]
for folder in folders:
    img_files = glob.glob(osp.join(input_dir, 'images', folder, f'*.{image_ext}'))
    
    _output_dir = osp.join(output_dir, folder)
    if not osp.exists(_output_dir):
        os.mkdir(_output_dir)
    
    for img_file in img_files:
        filename = osp.split(osp.splitext(img_file)[0])[-1]
        txt_file = osp.join(input_dir, 'labels', folder, filename + '.txt')
        assert osp.exists(txt_file), RuntimeError(f"There is no such label file for {txt_file}")

        image = cv2.imread(img_file)
        txt = open(txt_file, 'r')
        while True:
            line = txt.readline()
            if not line: break
            
            label_id, cx, cy, w, h = list(map(float, line.split(" ")))
            # print("xywh: ", cx, cy, w, h)
            xyxy = xywh2xyxy(image.shape, [cx, cy, w, h])
            # print("xyxy: ", xyxy)
            points, point = [], []
            for idx, __xyxy in enumerate(xyxy):
                if (idx)%2 == 0:
                    point = []
                    points.append(point)
                point.append(round(__xyxy, 1))

            x1, y1, x2, y2 = xyxy
            cv2.putText(image, str(label_id), (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 255), 1)
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

        cv2.imwrite(osp.join(_output_dir, filename + '.png'), image)

            
        
        