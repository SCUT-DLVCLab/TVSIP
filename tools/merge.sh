# 原始模型
modelPath=model/Qwen/Qwen2-5-VL-7B-Instruct
adapterModelPath=saves/tvsip_locator/checkpoint-510
CUDA_VISIBLE_DEVICES=1 llamafactory-cli export \
    --model_name_or_path $modelPath \
    --adapter_name_or_path $adapterModelPath \
    --template qwen2_vl \
    --finetuning_type lora \
    --export_dir saves/tvsip_locator/ep3 \
    --export_size 5 \
    --export_legacy_format False
