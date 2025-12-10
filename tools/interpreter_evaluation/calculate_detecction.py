import os
import re
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from PIL import Image
import argparse

def get_description(content):
    """
    Extracts the content between "The Tampered Areas" and "Judgment Basis" from the text.
    It attempts multiple regex patterns to handle variations in formatting (headers, bolding, etc.).
    """
    match = re.search(r"(### The Tampered Areas:(.*?)### Judgment Basis:)", content, re.DOTALL)
    match_end = match
    if not match: 
        match0 = re.search(r"### The Tampered Areas(.*?)### Judgment Basis", content, re.DOTALL)
        match_end = match0
        if not match0:
            match_0_1 = re.search(r"## The Tampered Areas(.*?)## Judgment Basis", content, re.DOTALL)
            match_end = match_0_1
            if not match_0_1:
                match_0_2 = re.search(r"## The Tampered Areas:(.*?)## Judgment Basis:", content, re.DOTALL)
                match_end = match_0_2
                if not match_0_2:
                    match0_0 = re.search(r"\*\*The Tampered Areas\*\*(.*?)\*\*Judgment Basis\*\*", content, re.DOTALL)
                    match_end = match0_0
                    if not match0_0:
                        match1 = re.search(r"\*\*The Tampered Areas:\*\*(.*?)\*\*Judgment Basis:\*\*", content, re.DOTALL)
                        match_end = match1
                        if not match1:
                            match2 = re.search(r"The Tampered Areas:(.*?)Judgment Basis:", content, re.DOTALL)
                            match_end = match2
                            if not match2:
                                match3 = re.search(r"The Tampered Areas(.*?)Judgment Basis", content, re.DOTALL)
                                match_end = match3
                                if not match3:
                                    return None
        
    # Extract the captured group containing the actual description text
    description_content = match_end.group(1)

    cleaned_content = description_content.strip()
    tampered_points = re.findall(r"^\d+\.\s", cleaned_content, re.MULTILINE)
    if len(tampered_points) == 0:
        return None

    match = re.search(r"(.*)(?=---)", cleaned_content, re.DOTALL)
    if match:
        cleaned_content = match.group(1)  # Extract content before "---"
        cleaned_content = cleaned_content.strip()

    return cleaned_content

def split_content_localiaztion(input_data):
    """
    Parses the extracted description to separate 'Content' and 'Location' fields.
    Returns formatted strings for both.
    """
    if input_data == 'None':
        return 'None', 'None'
    content_matches = re.findall(r'Content: (.+?)(?:\n|$)', input_data)
    location_matches = re.findall(r'Location: (.+?)(?:\n|$)', input_data)
    content_string = "\n".join([f"{i+1}.Content: \"{content}\"" for i, content in enumerate(content_matches)])
    location_string = "\n".join([f"{i+1}.Location: {location}" for i, location in enumerate(location_matches)])
    if content_string == '' or location_string == '':
        content_matches = re.findall(r'Content:\s*"([^"]+)"', input_data)
        location_matches = re.findall(r'Location:\s*(.+?)(?:\n|$)', input_data)
        content_string = "\n".join([f"{i+1}.Content: \"{content}\"" for i, content in enumerate(content_matches)])
        location_string = "\n".join([f"{i+1}.Location: {location}" for i, location in enumerate(location_matches)])
    return content_string, location_string


def is_tampered(file_content):
    """
    Determines if the file content indicates tampering.
    Logic: Checks if the "The Tampered Areas" section exists and contains valid numbered points.
    """
    tampered_section_match = get_description(file_content)
    if tampered_section_match == None or tampered_section_match.upper() == 'N/A':
        return False

    description_content, description_local = split_content_localiaztion(tampered_section_match)
    if description_content == '' and description_local == '':
        return False
    else:
        return True


def parse_file_label(file_name):
    """
    Determines the ground truth label based on the filename.
    Assumption: Files starting with "good" are authentic (not tampered).
    """
    if file_name.startswith("good"):
        return "not_tampered"
    return "tampered"

def main(folder_path):
    """
    Main function: Iterates through all txt files in the folder, compares predictions 
    against ground truth, and calculates performance metrics (Accuracy, Precision, Recall, F1).
    """
    gt_labels = []
    pred_labels = []

    for file_name in sorted(os.listdir(folder_path)):
        if file_name.endswith(".txt"):
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read()
            
            gt_label = parse_file_label(file_name)
            
            gt_labels.append(gt_label)
            
            pred_label = "tampered" if is_tampered(content) else "not_tampered"
            # print(file_name)
            # print(pred_label)
            pred_labels.append(pred_label)

    # Convert labels to binary format for metric calculation, 1 for "Tampered" (Positive), 0 for "Not Tampered" (Negative)
    gt_binary = [1 if label == "tampered" else 0 for label in gt_labels]
    pred_binary = [1 if label == "tampered" else 0 for label in pred_labels]

    # Calculate metrics
    precision = precision_score(gt_binary, pred_binary)
    recall = recall_score(gt_binary, pred_binary)
    f1 = f1_score(gt_binary, pred_binary)
    accuracy = accuracy_score(gt_binary, pred_binary)
    print('--------------------------------Detection Evaluation-------------------------------------------')
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process images with specified validation set")
    parser.add_argument('--val_set', type=str, default='Test', 
                        help='Validation set name')
    
    args = parser.parse_args()

    folder_path = f'results/Interpreter/{args.val_set}'
    main(folder_path)



