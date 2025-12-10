#Evaluate interpretability performance

VAL_SET="Test"

#1.Description
python tools/interpreter_evaluation/calculate_description.py --val_set $VAL_SET
#2.Detection
python tools/interpreter_evaluation/calculate_detecction.py --val_set $VAL_SET
#3.Localization
python tools/interpreter_evaluation/calculate_localization.py --val_set $VAL_SET
#4.Explanation
python tools/interpreter_evaluation/calculate_explanation.py --val_set $VAL_SET