from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
import os
from rouge_score import rouge_scorer
import argparse

def get_description(content):
    """
    Extracts the description of tampered areas from the text content.
    It attempts to match content between "The Tampered Areas" and "Judgment Basis"
    using various regex patterns to handle inconsistent formatting.
    """
    match= re.search(r"(### The Tampered Areas:(.*?)### Judgment Basis:)", content, re.DOTALL)
    match_end=match
    if not match:
        match0=re.search(r"### The Tampered Areas(.*?)### Judgment Basis", content, re.DOTALL)
        match_end=match0
        if not match0:
            match_0_1=re.search(r"## The Tampered Areas(.*?)## Judgment Basis", content, re.DOTALL)
            match_end=match_0_1
            if not match_0_1:
                match_0_2=re.search(r"## The Tampered Areas:(.*?)## Judgment Basis:", content, re.DOTALL)
                match_end=match_0_2
                if not match_0_2:
                    match0_0 = re.search(r"\*\*The Tampered Areas\*\*(.*?)\*\*Judgment Basis\*\*", content, re.DOTALL)
                    match_end=match0_0
                    if not match0_0:
                        match1 = re.search(r"\*\*The Tampered Areas:\*\*(.*?)\*\*Judgment Basis:\*\*", content, re.DOTALL)
                        match_end=match1
                        if not match1:
                            match2 = re.search(r"The Tampered Areas:(.*?)Judgment Basis:", content, re.DOTALL)
                            match_end=match2
                            if not match2:
                                match3 = re.search(r"The Tampered Areas(.*?)Judgment Basis", content, re.DOTALL)
                                match_end=match3
                                if not match3:
                                    return None
        
    description_content = match_end.group(1)

    cleaned_content=description_content.strip()
    tampered_points = re.findall(r"^\d+\.\s", cleaned_content, re.MULTILINE)
    if len(tampered_points) == 0:
        return None

    match = re.search(r"(.*)(?=---)", cleaned_content, re.DOTALL)
    if match:
        cleaned_content = match.group(1)
        cleaned_content=cleaned_content.strip()
    return cleaned_content


def split_content_localiaztion(input_data):
    """
    Parses the input string to separate 'Content' and 'Location' fields.
    Returns formatted strings for both.
    """
    if input_data=='None':
        return 'None','None'
    content_matches = re.findall(r'Content: (.+?)(?:\n|$)', input_data)
    location_matches = re.findall(r'Location: (.+?)(?:\n|$)', input_data)

    content_string = "\n".join([f"{i+1}.Content: \"{content}\"" for i, content in enumerate(content_matches)])
    location_string = "\n".join([f"{i+1}.Location: {location}" for i, location in enumerate(location_matches)])
    if content_string=='' or location_string=='':
        return 'None','None'

    return content_string,location_string

def calculate_rouge_l(txt1, txt2):
    """
    Calculates the ROUGE-L metric.
    
    Args:
    txt1: Text output from the large model (Prediction)
    txt2: Text from the Ground Truth (GT)
    
    Returns:
    The F-measure of the ROUGE-L score.
    """
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = scorer.score(txt2, txt1)
    rouge_l_scores = scores["rougeL"]
    return rouge_l_scores.fmeasure



def calculate_css(txt1, txt2):
    """
    Calculates the CSS (Cluster Similarity Score) / Cosine Similarity.
    
    Args:
    txt1: Text output from the large model
    txt2: Text from the Ground Truth
    
    Returns:
    The CSS value (Cosine Similarity).
    """
    vectorizer = CountVectorizer().fit([txt1, txt2])
    txt1_vector = vectorizer.transform([txt1]).toarray()
    txt2_vector = vectorizer.transform([txt2]).toarray()
    similarity = cosine_similarity(txt1_vector, txt2_vector)[0][0]
    css_score = similarity
    return css_score



def main(folder_path,gt_path):
    """
    Main function: Process all txt files in the folder and calculate evaluation metrics.
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
            description = get_description(content)
            gt_description=get_description(gt_content)
            # print(description)
            # print('-----------------gt----------------------------------')
            # print(gt_description)
            if description==None or description.upper()=='N/A':
                description='None'
            if gt_description==None:
                gt_description='None'

            description_content,description_local=split_content_localiaztion(description)
            gt_description_content,gt_description_local=split_content_localiaztion(gt_description)
            # print('------------------------------------split-------------------------')
            # print(description_content)
            # print(description_local)
            # print(gt_description_content)
            # print(gt_description_local)
            
            # css_score=calculate_css(description_content,gt_description_content)
            # rouge_l_score=calculate_rouge_l(description_content,gt_description_content)

            # css_score=calculate_css(description_local,gt_description_local)
            # rouge_l_score=calculate_rouge_l(description_local,gt_description_local)

            css_score=calculate_css(description,gt_description)
            rouge_l_score=calculate_rouge_l(description,gt_description)

            # if (description=='None' and gt_description=='None') or (description!='None' and gt_description!='None'):
            if (description!='None' and gt_description!='None'):
            # if (description!='None' or gt_description!='None'):
            # if 1:
                css_scores.append(css_score)
                rouge_l_scores.append(rouge_l_score)
            # print('--------------next_text-----------------------------')

    average_css = sum(css_scores) / len(css_scores)
    average_rougel=sum(rouge_l_scores)/ len(rouge_l_scores)
    print('--------------------------------Localization Evaluation-------------------------------------------')
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