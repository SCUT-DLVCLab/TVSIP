import os
import cv2
import numpy as np
import shutil
import argparse
from tqdm import tqdm

def is_contained(mask1, mask2):
    """
    Determine whether mask1 is completely enclosed by mask2
    """
    mask1_white = mask1 == 255
    mask2_white = mask2 == 255
    contained = np.logical_and(mask1_white, np.logical_not(mask2_white))
    return not np.any(contained)



def main(val_set):
    folder1 = f'results/Locator/LLVCB/{val_set}/segformer'
    folder2 = f'results/Locator/HLSCB/{val_set}/pred_w_BR'
    output_folder = f'results/Locator/Fusion/{val_set}/pred'

    os.makedirs(output_folder, exist_ok=True)

    file_list1 = os.listdir(folder1)

    for filename in tqdm(file_list1):
        file1_path = os.path.join(folder1, filename)
        file2_path = os.path.join(folder2, filename)
        output_path = os.path.join(output_folder, filename)
        if True:
            # Check if the corresponding file exists in folder 2
            if not os.path.exists(file2_path):
                shutil.copy(file1_path,output_path)
                continue
            img1 = cv2.imread(file1_path, cv2.IMREAD_GRAYSCALE)
            img2 = cv2.imread(file2_path, cv2.IMREAD_GRAYSCALE)

            # Because vision experts typically perform inference with padding as multiples of 8
            h, w = img2.shape
            img1 = img1[:h, :w]

            if True:
                avg_img = (img1.astype(np.float32) + img2.astype(np.float32)) / 2.0
                _, avg_img = cv2.threshold(avg_img, 127, 255, cv2.THRESH_BINARY)
                avg_img = avg_img.astype(np.uint8)

            if np.any(img1 != 0) and np.any(img2 != 0):
                if is_contained(img1,img2):
                    avg_img = img1
                    # print(f"contained：{output_path}")
            cv2.imwrite(output_path, avg_img)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process images with specified validation set")
    parser.add_argument('--val_set', type=str, default='Manual', 
                        help='Validation set name')
    
    args = parser.parse_args()
    main(args.val_set)