#!/bin/bash

VAL_SET="Test"
MODULE="Fusion"

python tools/OCR-based_Box_Rectification.py --val_set $VAL_SET
python tools/locator_fusion.py --val_set $VAL_SET
python tools/mask_highlight.py --val_set $VAL_SET

if [ "$VAL_SET" = "Test" ]; then
    python tools/locator_evaluation/eval_rtm.py --module $MODULE
fi

echo "All scripts completed!"