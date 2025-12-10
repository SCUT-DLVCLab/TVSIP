import os
import cv2
import numpy as np
from tqdm import tqdm
import argparse

def main(val_set):
    imgs_root = f'./datasets/TextDDLE/{val_set}/imgs'
    pred_root = f'results/Locator/Fusion/{val_set}/pred'
    ref_root = f'results/Locator/Fusion/{val_set}/ref'
    os.makedirs(ref_root, exist_ok=True)

    image_list = os.listdir(imgs_root)

    for i in tqdm(image_list):
        img_name = os.path.splitext(i)[0]
        img_p = os.path.join(imgs_root, i)
        img = cv2.imread(img_p)
        pred_p = os.path.join(pred_root, img_name + '.png')
        pred = (cv2.imread(pred_p,0)>=127).astype(np.uint8) * 255
        color_image = np.stack((pred,)*3, axis=-1)
        color_image[:,:,0] = 0
        color_image[:,:,2] = 0
        if img.size == color_image.size:
            highlighted_image = cv2.addWeighted(img, 0.5, color_image, 0.5, 0)
            img = np.where(np.stack((pred,)*3, axis=-1), highlighted_image,img)
            cv2.imwrite(os.path.join(ref_root, img_name+'.jpg'), img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process images with specified validation set")
    parser.add_argument('--val_set', type=str, default='Manual', 
                        help='Validation set name')
    
    args = parser.parse_args()
    main(args.val_set)
