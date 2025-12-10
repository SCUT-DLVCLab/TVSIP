from transformers import AutoProcessor,Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
import torch
import os
import torch.multiprocessing as mp
import argparse
import random
from tqdm import tqdm
import time

def split_list(lst, num_splits):
    random.shuffle(lst)
    total_length = len(lst)
    part_size = (total_length + num_splits - 1) // num_splits
    return [lst[i*part_size : (i+1)*part_size] for i in range(num_splits)]

def worker(gpu_id, test_subset, image_dir, ref_dir, explain_dir, model_dir):
    device = torch.device(f'cuda:{gpu_id}')
    # print(f"Process {os.getpid()} using GPU {gpu_id}, processing {len(test_subset)} items.")
    # all_time = 0

    MODEL_ID = model_dir
    min_pixels = 32*32
    max_pixels = 1536*28*28
    processor = AutoProcessor.from_pretrained(MODEL_ID, min_pixels=min_pixels, max_pixels=max_pixels, use_fast =False)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype="auto",
        attn_implementation="flash_attention_2"
    ).to(device).eval()

    system_prompt = "As a tampered text detection expert, you are capable of describing text images, analyzing the authenticity of images, locating areas where text has been tampered with, and providing your judgments based on both high-level semantic clues and low-level visual clues."
    for i in tqdm(test_subset):
        if True:
            im_p = os.path.join(image_dir, i + ".jpg")
            ref_p = os.path.join(ref_dir, i + ".jpg")
            messages = [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": system_prompt},
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": im_p,
                        },
                        {
                            "type": "image",
                            "image": ref_p,
                        },
                        {"type": "text", "text": "Please describe and analyze this image. Refer to the potentially green-highlighted suspicious areas from another image to assess the authenticity of the image."},
                    ],
                }
            ]
            # start_time = time.time()
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(device)

            generated_ids = model.generate(**inputs, max_new_tokens=4096)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            save1 = os.path.join(explain_dir, i + ".txt")
            with open(save1, 'w', encoding='utf-8') as file:
                file.write(output_text)
            # end_time = time.time()
            # t = end_time - start_time
            # all_time += t
    # print(all_time)

def main(val_set, gpu_select):
    num_processes = len(gpu_select)

    image_dir = f'./datasets/TextDDLE/{val_set}/imgs'
    ref_dir = f'results/Locator/Fusion/{val_set}/ref'
    explain_dir = f'results/Interpreter/{val_set}'
    model_dir = 'saves/tvsip_interpreter_stage2/ep3'
    os.makedirs(explain_dir, exist_ok=True)

    test_list = [os.path.splitext(i)[0] for i in os.listdir(image_dir)]
    test_subsets = split_list(test_list, num_processes)

    processes = []
    
    for idx, gpu_id in enumerate(gpu_select):
        p = mp.Process(target=worker, args=(gpu_id, test_subsets[idx], image_dir, ref_dir, explain_dir, model_dir))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process images with specified validation set")
    parser.add_argument('--val_set', type=str, default='Test', 
                        help='Validation set name')
    parser.add_argument('--gpus', type=int, nargs='+', default=[2], 
                        help='GPU IDs to use (e.g., --gpus 0 1 2 3)')
    args = parser.parse_args()
    mp.set_start_method('spawn')
    main(args.val_set, args.gpus)