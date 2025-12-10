from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
import os
from rouge_score import rouge_scorer
import argparse


def extract_clues_as_string(content):
    """
    Extracts the description text from the content using various regex patterns.
    It attempts to handle different formatting styles (Markdown headers, bold, plain text).
    """

    # Attempt to match the "Judgment Basis" block from the text
    judgment_match = re.search(r"Judgment Basis\s*(.*)", content, re.DOTALL)

    # If "Judgment Basis" is found, attempt to directly extract points 1 and 2 from it
    if judgment_match:
        judgment_content = judgment_match.group(1)

        # Extract "1. Low-level Visual Clues" from the Judgment Basis section
        point_1_match = re.search(
            r"1\.\s*Low-level Visual Clues\s*(.*?)(?=2\.|$)",
            judgment_content,
            re.DOTALL
        )
        # Extract "2. High-level Semantic Clues" from the Judgment Basis section
        point_2_match = re.search(
            r"2\.\s*High-level Semantic Clues\s*(.*)",
            judgment_content,
            re.DOTALL
        )

        point_1 = point_1_match.group(1).strip() if point_1_match else "None"
        point_2 = point_2_match.group(1).strip() if point_2_match else "None"

        # If the required content is successfully matched in Judgment Basis, return it directly
        if point_1!='None' or point_2!='None':
            result = "1. Low-level Visual Clues:\n"
            result += f"   {point_1}\n\n"
            result += "2. High-level Semantic Clues:\n"
            result += f"   {point_2}"
            return result
        else:
            pass

    # Fallback pattern for Low-level Visual Clues
    low_level_pattern = r"Low-level\s*Visual\s*Clues\s*:\s*(.*?)(?=\n[ \t]*[A-Z][^\n]*?:|$)"

    low_level_match = re.search(low_level_pattern, content, re.DOTALL)
    low_level_clues = None
    if low_level_match:
        low_level_block = low_level_match.group(1).strip()
        bullet_points = re.findall(r"-\s*(.*)", low_level_block)
        if bullet_points:
            low_level_clues = "\n".join([f"   - {bp.strip()}" for bp in bullet_points])
        else:
            low_level_clues = "   " + low_level_block

    # Extract the "High-level Semantic Clues" section
    high_level_pattern = r"High-level\s*Semantic\s*Clues\s*:\s*(.*?)(?=\n[ \t]*[A-Z][^\n]*?:|$)"
    high_level_match = re.search(high_level_pattern, content, re.DOTALL)
    high_level_clues = None
    if high_level_match:
        high_level_block = high_level_match.group(1).strip()
        bullet_points = re.findall(r"-\s*(.*)", high_level_block)
        if bullet_points:
            high_level_clues = "\n".join([f"   - {bp.strip()}" for bp in bullet_points])
        else:
            high_level_clues = "   " + high_level_block

    if not low_level_clues and not high_level_clues:
        return None

    # Assemble the final result
    result = ""
    if low_level_clues:
        result += "1. Low-level Visual Clues:\n" + low_level_clues + "\n\n"
    else:
        return None

    if high_level_clues:
        result += "2. High-level Semantic Clues:\n" + high_level_clues
    else:
        result += "2. High-level Semantic Clues:\n   None"

    return result



def calculate_rouge_l(txt1, txt2):
    """
    Calculates the ROUGE-L metric.
    
    Args:
    txt1: Text string output by the large model.
    txt2: Text string of the Ground Truth (GT).
    
    Returns:
    The F-measure of the ROUGE-L score.
    """
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = scorer.score(txt2, txt1)  # 注意：GT 在前，模型输出在后
    rouge_l_scores = scores["rougeL"]
    return rouge_l_scores.fmeasure



def calculate_css(txt1, txt2):
    """
    Calculates the CSS (Cluster Similarity Score / Cosine Similarity).
    
    Args:
    txt1: Text string output by the large model.
    txt2: Text string of the Ground Truth (GT).
    
    Returns:
    The Cosine Similarity score.
    """
    vectorizer = CountVectorizer().fit([txt1, txt2])
    txt1_vector = vectorizer.transform([txt1]).toarray()
    txt2_vector = vectorizer.transform([txt2]).toarray()
    similarity = cosine_similarity(txt1_vector, txt2_vector)[0][0]
    css_score = similarity
    return css_score


def main(folder_path,gt_path):
    """
    Main function: Process all txt files in the folder and calculate metrics.
    """
    gt_labels = []
    pred_labels = []
    css_scores = []
    rouge_l_scores=[]

    for file_name in sorted(os.listdir(folder_path)):
        if file_name.endswith(".txt"):
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read()
            gt_file_path=os.path.join(gt_path,file_name)
            with open(gt_file_path, "r", encoding="utf-8") as gt_file:
                gt_content = gt_file.read()
            
            # print(file_name)
            description = extract_clues_as_string(content)
            gt_description=extract_clues_as_string(gt_content)
            # print(description)
            # print('---------gt---------')
            # print(gt_description)

            if description==None:
                description='None'
            if gt_description==None:
                gt_description='None'
            # if (description=='None' and gt_description=='None') or (description!='None' and gt_description!='None'):
            if (description!='None' and gt_description!='None'):
            # if (description!='None' or gt_description!='None'):
            # if 1:
                css_score=calculate_css(description,gt_description)
                rouge_l_score=calculate_rouge_l(description,gt_description)
                css_scores.append(css_score)
                rouge_l_scores.append(rouge_l_score)
            # print('--------------------------------next_txt-------------------------------------------')

    average_css = sum(css_scores) / len(css_scores)
    average_rougel=sum(rouge_l_scores)/ len(rouge_l_scores)
    print('--------------------------------Explanation Evaluation-------------------------------------------')
    print('Average CSS is', average_css)
    print('Average RougeL is', average_rougel)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process images with specified validation set")
    parser.add_argument('--val_set', type=str, default='Test', 
                        help='Validation set name')
    
    args = parser.parse_args()

    folder_path = f'results/Interpreter/{args.val_set}'
    gt_path = f"./datasets/TextDDLE/{args.val_set}/explain"
    main(folder_path,gt_path)