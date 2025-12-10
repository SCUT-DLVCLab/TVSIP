import os
import json
from tqdm import tqdm
from shapely.geometry import Polygon
from difflib import SequenceMatcher
import cv2
import numpy as np
import argparse

def read_json(path):
    with open(path, "r") as file:
        data = json.load(file)
    return data

def read_txt(path):
    with open(path, "r") as file:
        content = file.read()
    return content

def parse_bbox_coordinates(coord_str, height, width):
    coord = json.loads(coord_str.replace("```json", "").replace("```", "").strip())
    for i, c in enumerate(coord):
        x1, y1, x2, y2 = c['bbox_2d']
        x1 = int(x1 / 1344 * width)
        x2 = int(x2 / 1344 * width)
        y1 = int(y1 / 1344 * height)
        y2 = int(y2 / 1344 * height)
        x1 = max(0, min(x1, width - 1))
        x2 = max(0, min(x2, width - 1))
        y1 = max(0, min(y1, height - 1))
        y2 = max(0, min(y2, height - 1))
        c['bbox_2d'] = [x1, y1, x2, y2]
        coord[i] = c
    return coord

def bbox_from_position_list(position_list):
    """[left, top, right, bottom]"""
    x_coords = position_list[::2]
    y_coords = position_list[1::2]
    return [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]  # [left, top, right, bottom]

def compute_iou(bbox1, bbox2):
    x_left = max(bbox1[0], bbox2[0])
    y_top = max(bbox1[1], bbox2[1])
    x_right = min(bbox1[2], bbox2[2])
    y_bottom = min(bbox1[3], bbox2[3])

    if x_right <= x_left or y_bottom <= y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    bbox1_area = (bbox1[2]-bbox1[0]) * (bbox1[3]-bbox1[1])
    bbox2_area = (bbox2[2]-bbox2[0]) * (bbox2[3]-bbox2[1])

    union_area = bbox1_area + bbox2_area - intersection_area

    if union_area == 0:
        return 0.0

    iou = intersection_area / union_area
    return iou

def compute_center_distance(box1, box2):
    center_box1 = ((box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2)
    center_box2 = ((box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2)
    distance = np.sqrt((center_box1[0] - center_box2[0]) ** 2 + (center_box1[1] - center_box2[1]) ** 2)
    return distance

def compute_content_similarity(text1, text2):
    return SequenceMatcher(None, text1, text2).ratio()

def extract_approximate_word_positions(line_text, char_details, target_word='whas', similarity_threshold=0.8):
    """
    Extract all approximate occurrences of the target word in a line of text and their bounding boxes.

    Args:
        line_text (str): Line text content, e.g., "high what app is what"
        char_details (list): List of position information for each character, formatted as [{"h": [x1, y1, x2, y2]}, ...]
        target_word (str): Target word, e.g., "what"
        similarity_threshold (float): Similarity threshold, e.g., 0.8

    Returns:
        list: List of bounding boxes for each match, formatted as [[x1, y1, x2, y2], ...]
    """
    positions = []
    target_length = len(target_word)
    min_window = max(1, target_length - 1)  # Minimum window size to avoid negative values
    max_window = target_length + 1          # Maximum window size

    for window_size in range(min_window, max_window + 1):
        for start_idx in range(len(line_text) - window_size + 1):
            end_idx = start_idx + window_size
            substring = line_text[start_idx:end_idx]
            similarity = SequenceMatcher(None, target_word, substring).ratio()
            
            if similarity >= similarity_threshold:
                # Extract position information for the corresponding characters
                matched_char_positions = []
                for i in range(start_idx, end_idx):
                    if i < len(char_details):
                        char_info = char_details[i]
                        for char, pos in char_info.items():
                            matched_char_positions.extend(pos)
                    else:
                        print(f"Warning: Character index {i} is out of range for char_details.")
                
                if matched_char_positions:
                    x_coords = matched_char_positions[::2]
                    y_coords = matched_char_positions[1::2]
                    bbox = [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]
                    positions.append({
                        'content': substring,
                        'similarity': similarity,
                        'bbox': bbox
                    })
                else:
                    print(f"Warning: Character position information not found for indices {start_idx} to {end_idx}.")
    
    return non_overlapping_matching(positions)

def non_overlapping_matching(matches, overlap_threshold=0.1):
    if not matches:
        return []
    sorted_matches = sorted(matches, key=lambda x: x['similarity'], reverse=True)
    selected_matches = []

    for match in sorted_matches:
        keep = True
        match_bbox = match['bbox']
        for selected in selected_matches:
            selected_bbox = selected['bbox']
            iou = compute_iou(match_bbox, selected_bbox)
            if iou > overlap_threshold:
                keep = False
                break
        if keep:
            selected_matches.append(match)

    return selected_matches

def select_best_box(target_box, candidate):
    """
    Select the most appropriate box from the candidate boxes

    Parameters:
    - target_box: list or array of [x1, y1, x2, y2]
    - candidate_boxes: list of lists or arrays, each element is [x1, y1, x2, y2]

    Returns:
    - best_box: list of [x1, y1, x2, y2]
    """
    overlapping_condi = []
    ious = []
    for m in candidate:
        box = m['bbox']
        iou = compute_iou(target_box, box)
        if iou > 0:
            overlapping_condi.append(m)
            ious.append(iou)

    if ious:
        # If there are overlapping boxes, select the one with the largest IoU
        max_iou_index = np.argmax(ious)
        best = overlapping_condi[max_iou_index]
        iou = ious[max_iou_index]
        similarity = best['similarity']
        best_match = {
                'similarity': similarity,
                'iou':iou,
                'content':best['content'],
                'bbox': best['bbox']
            }
    else:
        # If there are no overlapping boxes, select the closest one
        distances = [compute_center_distance(target_box, c['bbox']) for c in candidate]
        min_distance_index = np.argmin(distances)
        best = candidate[min_distance_index]
        iou = 0
        similarity = best['similarity']
        score = 0.5 * iou + 0.5 * similarity
        best_match = {
                'score': score,
                'similarity': similarity,
                'iou':iou,
                'content':best['content'],
                'bbox': best['bbox']
            }
    return best_match

def match_targets_with_ocr(targets, ocr_results, content_similarity_threshold=0.0):
    """
    First retrieve all OCR results with similar content, then compare their bounding boxes to find the best match.
    """
    matches = []

    for target in targets:
        content_extract = []
        target_bbox = target['bbox_2d']  # [left, top, right, bottom]
        target_content = target['content']
        candidate_matches = []

        # Step 1: Retrieve all content-similar OCR results
        for ocr_item in ocr_results:
            for line_text, line_content in ocr_item.items():
                position_list = line_content['position']
                line_bbox = bbox_from_position_list(position_list)

                # Calculate content similarity
                content_similarity = compute_content_similarity(target_content, line_text)

                # # If content similarity is below the threshold, ignore this match
                if content_similarity <= content_similarity_threshold:
                    continue

                # Add similar content to the candidate list
                candidate_matches.append({
                    'text': line_text,
                    'bbox': line_bbox,
                    'char_details': line_content.get('char_details', [])
                })

        # If no content-similar matches found, skip
        if not candidate_matches:
            matches.append({
                'target': target,
                'best_match': None
            })
            continue
        
        # Step 2: Filter content-similar candidates based on bbox similarity
        for candidate in candidate_matches:
            # Find the position of the search target in the matched line (bounding box of the substring)
            sub_content = extract_approximate_word_positions(candidate['text'], candidate['char_details'], target_content, similarity_threshold=0.8)
            content_extract.extend(sub_content)
        if len(content_extract) == 1:
            m = content_extract[0]
            iou = compute_iou(target_bbox,m['bbox'])
            similarity = m['similarity']
            best_match = {
                    'similarity': similarity,
                    'iou':iou,
                    'content':m['content'],
                    'bbox': m['bbox']
                }
        elif len(content_extract)==0:
            best_match = None
        else:
            best_match = select_best_box(target_bbox, content_extract)

        matches.append({
            'target': target,
            'best_match': best_match
        })

    return matches


def is_large_area_enclosed(box_a, box_b, threshold=0.8):
    """
    Determine whether bounding box A is largely enclosed by bounding box B.
    
    Parameters:
    - box_a: tuple，Coordinates of bounding box A (x1_A, y1_A, x2_A, y2_A)
    - box_b: tuple，Coordinates of bounding box B (x1_B, y1_B, x2_B, y2_B)
    - threshold: float，Threshold for substantial enclosure, default is 0.8 (80%)
    
    return:
    - True: If the area of bounding box A enclosed by bounding box B exceeds the threshold
    """
    x1_A, y1_A, x2_A, y2_A = box_a
    x1_B, y1_B, x2_B, y2_B = box_b
    
    area_A = (x2_A - x1_A) * (y2_A - y1_A)
    if area_A <= 0:
        return False
    
    inter_x1 = max(x1_A, x1_B)
    inter_y1 = max(y1_A, y1_B)
    inter_x2 = min(x2_A, x2_B)
    inter_y2 = min(y2_A, y2_B)
    
    # If there is no intersection, return False directly
    if inter_x1 >= inter_x2 or inter_y1 >= inter_y2:
        return False
    intersection_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    return (intersection_area / area_A) > threshold

def expand_box(box, scale, img_width, img_height):
    """
    Expand the bounding box proportionally.
    
    参数:
    - box: tuple，coordinates of the bounding box (x1, y1, x2, y2)
    - scale: float，expansion ratio (e.g., 0.2 means expand by 20%)
    """
    x1, y1, x2, y2 = box
    width = x2 - x1
    height = y2 - y1

    expand_w = scale * width
    expand_h = scale * height

    new_x1 = x1 - expand_w / 2
    new_y1 = y1 - expand_h / 2
    new_x2 = x2 + expand_w / 2
    new_y2 = y2 + expand_h / 2

    new_x1 = max(0, new_x1)
    new_y1 = max(0, new_y1)
    new_x2 = min(img_width, new_x2)
    new_y2 = min(img_height, new_y2)
    
    return (int(new_x1), int(new_y1), int(new_x2), int(new_y2))

def main(val_set):
    ocr_json_dir = f'./datasets/TextDDLE/{val_set}/ocr'
    HLSCB_dir = f'results/Locator/HLSCB/{val_set}/query3'
    LLVCB_dir = f'results/Locator/LLVCB/{val_set}/segformer' # provide the shape of images
    fusion_dir = f'results/Locator/HLSCB/{val_set}/pred_w_BR'

    query_list = os.listdir(HLSCB_dir)
    all_list = sorted(os.listdir(LLVCB_dir))

    os.makedirs(fusion_dir, exist_ok=True)

    for name_ in tqdm(all_list, desc="Processing images"):
        name = os.path.splitext(name_)[0]
        # tamper or not
        if (name + '.txt') in query_list:
            ocr_json_path = f'{ocr_json_dir}/{name}.json'
            hlscb_path = f'{HLSCB_dir}/{name}.txt'
            llvcb_path = f'{LLVCB_dir}/{name}.png'
            fusion_path = f'{fusion_dir}/{name}.png'

            height, width = cv2.imread(llvcb_path).shape[:2]
            pred = np.zeros((height, width), dtype=np.uint8)

            ocr = read_json(ocr_json_path)
            print(f"Processing OCR JSON: {ocr_json_path}")
            print(f"Bounding Box and Content Path: {hlscb_path}")
            print(f"Reference Path: {llvcb_path}")

            bbox_and_content = parse_bbox_coordinates(read_txt(hlscb_path), height, width)

            matches = match_targets_with_ocr(bbox_and_content, ocr, content_similarity_threshold=0.0)

            for match in matches:
                target = match['target']
                best_match = match['best_match']
                print("Retrieval Target:", target)
                if best_match:
                    bbox = best_match['bbox']
                    iou = best_match['iou']
                    if bbox:
                        if is_large_area_enclosed(bbox,target['bbox_2d'],0.9) and iou > 0.5:
                            bbox = target['bbox_2d']
                        else:
                            bbox = expand_box(bbox, 0.2, width,height)
                        x1, y1, x2, y2 = bbox
                        cv2.rectangle(pred, (x1, y1), (x2, y2), color=255, thickness=-1)
                        similarity = best_match['similarity']
                        content = best_match['content']
                        print("Best Matching Text：", content)
                        print("Content Similarity:", similarity)
                        print("Position Similarity (IoU):", iou)
                        print("Retrieved Content:", content)
                    else:
                        print("No valid substring match found")
                else:
                    bbox = target['bbox_2d']
                    x1, y1, x2, y2 = bbox
                    cv2.rectangle(pred, (x1, y1), (x2, y2), color=255, thickness=-1)
                    print("No matching results found")

                print("-" * 50)
            cv2.imwrite(fusion_path, pred)
        else:
            llvcb_path = f'{LLVCB_dir}/{name}.png'
            fusion_path = f'{fusion_dir}/{name}.png'
            height, width = cv2.imread(llvcb_path).shape[:2]
            pred = np.zeros((height, width), dtype=np.uint8)
            cv2.imwrite(fusion_path, pred)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process images with specified validation set")
    parser.add_argument('--val_set', type=str, default='Manual', 
                        help='Validation set name')
    
    args = parser.parse_args()
    main(args.val_set)