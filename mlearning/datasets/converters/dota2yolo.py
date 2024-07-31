from pathlib import Path
import cv2 
import os
import os.path as osp
from tqdm import tqdm
from shutil import copyfile
VERBOSE = str(os.getenv("YOLO_VERBOSE", True)).lower() == "true"  # global verbose mode
TQDM_BAR_FORMAT = "{l_bar}{bar:10}{r_bar}" if VERBOSE else None  # tqdm bar format


def convert_dota2yolo_obb(dota_root_path: str, save_dir: str, 
                          image_width: int=None, image_height: int=None,
                          class2idx: dict={}, 
                          copy_image: bool=True, image_ext='bmp'):

    dota_root_path = Path(dota_root_path)
    def convert_label(image_name, image_width, image_height, orig_label_dir, save_dir):
        """Converts a single image's DOTA annotation to YOLO OBB format and saves it to a specified directory."""
        orig_label_path = orig_label_dir / f"{image_name}.txt"
        save_path = save_dir / f"{image_name}.txt"

        with orig_label_path.open("r") as f, save_path.open("w") as g:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 9:
                    continue
                class_name = parts[8]
                if class_name in class2idx:
                    class_idx = class2idx[class_name]
                else:
                    class_idx = len(class2idx)
                    class2idx[class_name] = class_idx
                coords = [float(p) for p in parts[:8]]
                normalized_coords = [
                    coords[i] / image_width if i % 2 == 0 else coords[i] / image_height for i in range(8)
                ]
                formatted_coords = ["{:.6g}".format(coord) for coord in normalized_coords]
                g.write(f"{class_idx} {' '.join(formatted_coords)}\n")

    for phase in ["train", "val"]:
        if copy_image:
            image_dir = dota_root_path  / phase/ "images"
            orig_label_dir = dota_root_path / phase / "labelTxt" 
            save_labels_dir = Path(save_dir) / "labels" / phase
            save_labels_dir.mkdir(parents=True, exist_ok=True)
            save_images_dir = Path(save_dir) / "images" / phase
            save_images_dir.mkdir(parents=True, exist_ok=True)
            image_paths = list(image_dir.iterdir())
            for image_path in tqdm(image_paths, desc=phase):
                image_name_without_ext = image_path.stem
                copyfile(image_path, save_images_dir / Path(image_path.stem + image_path.suffix))
                img = cv2.imread(str(image_path))
                h, w = img.shape[:2]
                convert_label(image_name_without_ext, w, h, orig_label_dir, save_labels_dir)
        else:
            assert image_width is not None and image_height is not None, ValueError(f"Height({image_height}) and Width({image_width}) of image must not be None")
            orig_label_dir = dota_root_path / phase / "labelTxt" 
            save_labels_dir = Path(save_dir) / "labels" / phase
            save_labels_dir.mkdir(parents=True, exist_ok=True)
            label_paths = list(orig_label_dir.iterdir())
            for label_path in tqdm(label_paths, desc=phase):
                image_name_without_ext = label_path.stem
                h, w = img.shape[:2]
                convert_label(image_name_without_ext, image_width, image_height, orig_label_dir, save_labels_dir)


    idx2class_txt = open(osp.join(save_dir, 'idx2class.txt'), 'w')
    class2idx_txt = open(osp.join(save_dir, 'class2idx.txt'), 'w')
    for key, val in class2idx.items():
        idx2class_txt.write(f"{val}: {key}\n")
        class2idx_txt.write(f"{key}: {val}\n")
        
    idx2class_txt.close()
    class2idx_txt.close()
