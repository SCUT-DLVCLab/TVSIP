import os
import cv2
import numpy as np
import re
import json
import argparse

def main(val_set):
    hlscb_root = f'results/Locator/HLSCB/{val_set}/query3'
    llvcb_root = f'results/Locator/LLVCB/{val_set}/segformer'
    pred_dir = f'results/Locator/HLSCB/{val_set}/pred_wo_BR'

    os.makedirs(pred_dir, exist_ok=True)

    def parse_bbox_coordinates(coord_str):
        coord = json.loads(coord_str.replace("```json", "").replace("```", "").strip())
        bboxes = [i["bbox_2d"] for i in coord]
        return bboxes

    image_list = [os.path.splitext(i)[0] for i in os.listdir(llvcb_root)]
    for image_name in image_list:
        img_file_name = image_name + '.png'
        img_path = os.path.join(llvcb_root, img_file_name)
        pred_path = os.path.join(pred_dir, image_name + '.png')
        
        if not os.path.exists(img_path):
            print(f"Image file {img_path} does not exist.")
            continue
        
        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to read image {img_path}.")
            continue
        height, width = img.shape[:2]
        
        mask = np.zeros((height, width), dtype=np.uint8)
        
        txt_file_name = image_name + '.txt'
        txt_path = os.path.join(hlscb_root, txt_file_name)
        print(txt_path)
        
        if os.path.exists(txt_path):
            with open(txt_path, 'r', encoding='utf-8') as f:
                coord_str = f.read()
            
            bboxes = parse_bbox_coordinates(coord_str)

            if bboxes:
                for bbox in bboxes:
                    x1, y1, x2, y2 = bbox
                    x1 = int(x1 / 1344 * width)
                    x2 = int(x2 / 1344 * width)
                    y1 = int(y1 / 1344 * height)
                    y2 = int(y2 / 1344 * height)

                    x1 = max(0, min(x1, width - 1))
                    x2 = max(0, min(x2, width - 1))
                    y1 = max(0, min(y1, height - 1))
                    y2 = max(0, min(y2, height - 1))
        
                    cv2.rectangle(mask, (x1, y1), (x2, y2), color=255, thickness=-1)
            else:
                pass
        cv2.imwrite(pred_path, mask)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process images with specified validation set")
    parser.add_argument('--val_set', type=str, default='Manual', 
                        help='Validation set name')
    
    args = parser.parse_args()
    main(args.val_set)