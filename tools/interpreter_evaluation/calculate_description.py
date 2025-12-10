from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
import os
from rouge_score import rouge_scorer
import argparse


def get_description(content):
    """
    Extracts the description text from the content using various regex patterns.
    It attempts to handle different formatting styles (Markdown headers, bold, plain text).
    """
    match= re.search(r"(### Description:(.*?)### Detection:)", content, re.DOTALL)
    match_end=match
    if not match:
        match0=re.search(r"### Description(.*?)### Detection", content, re.DOTALL)
        match_end=match0
        if not match0:
            match_0_1=re.search(r"## Description(.*?)## Detection", content, re.DOTALL)
            match_end=match_0_1
            if not match_0_1:
                match_0_2=re.search(r"## Description:(.*?)## Detection:", content, re.DOTALL)
                match_end=match_0_2
                if not match_0_2:
                    match0_0 = re.search(r"\*\*Description\*\*(.*?)\*\*Detection\*\*", content, re.DOTALL)
                    match_end=match0_0
                    if not match0_0:
                        match1 = re.search(r"\*\*Description:\*\*(.*?)\*\*Detection:\*\*", content, re.DOTALL)
                        match_end=match1
                        if not match1:
                            match2 = re.search(r"Description:(.*?)Detection:", content, re.DOTALL)
                            match_end=match2
                            if not match2:
                                match3 = re.search(r"Description(.*?)Detection", content, re.DOTALL)
                                match_end=match3
                                if not match3:
                                    return None
        
    description_content = match_end.group(1)
    cleaned_content=description_content.strip()
    match = re.search(r"(.*)(?=---)", cleaned_content, re.DOTALL)
    if match:
        cleaned_content = match.group(1)
        cleaned_content=cleaned_content.strip()
    return cleaned_content

def calculate_rouge_l(txt1, txt2):
    """
    Calculates the ROUGE-L metric.
    
    Args:
        txt1: Text output from the large model (Prediction).
        txt2: Ground Truth (GT) text (Reference).
    
    Returns:
        The F-measure of the ROUGE-L score.
    """
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = scorer.score(txt2, txt1)  # 注意：GT 在前，模型输出在后
    rouge_l_scores = scores["rougeL"]
    return rouge_l_scores.fmeasure

def calculate_css(txt1, txt2):
    """
    Calculates the Cluster Similarity Score (CSS) using Cosine Similarity.
    
    Args:
        txt1: Text output from the large model.
        txt2: Ground Truth (GT) text.
    
    Returns:
        The cosine similarity score.
    """
    vectorizer = CountVectorizer().fit([txt1, txt2])
    txt1_vector = vectorizer.transform([txt1]).toarray()
    txt2_vector = vectorizer.transform([txt2]).toarray()
    similarity = cosine_similarity(txt1_vector, txt2_vector)[0][0]
    css_score = similarity

    return css_score

def main(folder_path,gt_path):
    """
    Main function: Process all .txt files in the folder and calculate average metrics.
    """
    gt_labels = []
    pred_labels = []
    css_scores = []
    rouge_l_scores=[]

    for file_name in os.listdir(folder_path):
        if file_name.endswith(".txt"):
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read()

            gt_file_path=os.path.join(gt_path,file_name)
            with open(gt_file_path, "r", encoding="utf-8") as gt_file:
                gt_content = gt_file.read()

            description = get_description(content)
            gt_description=get_description(gt_content)

            if description==None or gt_description==None:
                continue
            else:
                pass
            css_score=calculate_css(description,gt_description)
            rouge_l_score=calculate_rouge_l(description,gt_description)

            css_scores.append(css_score)
            rouge_l_scores.append(rouge_l_score)

    average_css = sum(css_scores) / len(css_scores)
    average_rougel=sum(rouge_l_scores)/ len(rouge_l_scores)
    print('--------------------------------Description Evaluation-------------------------------------------')
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