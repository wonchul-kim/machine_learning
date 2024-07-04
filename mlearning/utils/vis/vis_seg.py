import os.path as osp
import cv2 
import numpy as np

def vis_seg(img_file, idx2masks, idx2class, output_dir, color_map):
    
    filename = osp.split(osp.splitext(img_file)[0])[-1]
    img = cv2.imread(img_file)
    height, width, channel = img.shape
    vis_mask = np.zeros((height, width, channel))
    
    origin = 25, 25
    font = cv2.FONT_HERSHEY_SIMPLEX
    text1 = np.zeros((50, width, 3), np.uint8)
    text2 = np.zeros((50, width, 3), np.uint8)
    text3 = np.zeros((50, 300, 3), np.uint8)
    cv2.putText(text1, "(a) original", origin, font, 0.6, (255, 255, 255), 1)
    cv2.putText(text2, "(b) predicted", origin, font, 0.6, (255, 255, 255), 1)
    cv2.putText(text3, "(c) legend", origin, font, 0.6, (255, 255, 255), 1)
    
    for cls, masks in idx2masks.items():
        for mask in masks:
            points = np.array(mask, dtype=np.int32)
            cv2.fillConvexPoly(vis_mask, points, color=tuple(map(int, color_map[int(cls)])))
            # cv2.fillPoly(vis_mask, [points], color=tuple(map(int, color_map[int(cls)])))

    vis_img = cv2.vconcat([text1, img])

    vis_mask = vis_mask.astype(np.uint8)
    vis_mask = cv2.addWeighted(img, 0.4, vis_mask, 0.6, 0)
    vis_mask = cv2.vconcat([text2, vis_mask])

    vis_legend = np.zeros((height, 300, channel), dtype="uint8")
    for idx, color in enumerate(color_map):
        color = [int(c) for c in color]
        cv2.putText(vis_legend, idx2class[idx], (5, (idx * 25) + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.rectangle(vis_legend, (150, (idx * 25)), (300, (idx * 25) + 25), tuple(color), -1)
    vis_legend = cv2.vconcat([text3, vis_legend])

    vis_res = cv2.hconcat([vis_img, vis_mask, vis_legend])

    cv2.imwrite(osp.join(output_dir, filename + '.png'), vis_res)