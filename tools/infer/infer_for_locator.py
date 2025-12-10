from transformers import Qwen2VLForConditionalGeneration, AutoProcessor,Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
import torch
import os
import torch.multiprocessing as mp
import argparse
import random
from tqdm import tqdm
from PIL import Image

def split_list(lst, num_splits):
    random.shuffle(lst)
    total_length = len(lst)
    part_size = (total_length + num_splits - 1) // num_splits
    return [lst[i*part_size : (i+1)*part_size] for i in range(num_splits)]

def resize_image(image_path, target_size=(1344, 1344)):
    """
    Resize image to target size
    """
    img = Image.open(image_path)
    img_resized = img.resize(target_size, Image.LANCZOS)
    return img_resized


def worker(gpu_id, test_subset, image_dir, results_dir, model_dir):
    device = torch.device(f'cuda:{gpu_id}')
    # print(f"Process {os.getpid()} using GPU {gpu_id}, processing {len(test_subset)} items.")

    MODEL_ID = model_dir
    min_pixels = 32*32
    max_pixels = 2000*2000
    processor = AutoProcessor.from_pretrained(MODEL_ID, min_pixels=min_pixels, max_pixels=max_pixels, use_fast =False)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype="auto",
        attn_implementation="flash_attention_2"
    ).to(device).eval()

    system_prompt = "As a tampered text detection expert, you are capable of observing high-level semantic clues in the text within an image to determine whether the image has been tampered with. This includes contextual inconsistencies, logical implausibility, grammatical or spelling errors, key-value mismatch, and other discrepancies."
    for i in tqdm(test_subset):
        if True:
            im_p = os.path.join(image_dir, i + ".jpg")
            # print(f"GPU {gpu_id} processing: {im_p}")
            img_resized = resize_image(im_p, target_size=(1344, 1344))
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
                            "image": img_resized,
                        },
                        {"type": "text", "text": "Is the image authentic or has it been tampered with?"},
                    ],
                }
            ]
            for num in range(len(results_dir)):
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
                if num == 0:
                    save1 = os.path.join(results_dir[num], i + ".txt")
                    with open(save1, 'w', encoding='utf-8') as file:
                        file.write(output_text)

                    query = 'Please locate the tampered areas and describe their tampered content and positions.'
                    new_message = [
                        {"role": "assistant", "content": [{"type": "text", "text": output_text}]},
                        {"role": "user", "content": [{"type": "text", "text": query}]}
                                   ]
                if num == 1:
                    save2 = os.path.join(results_dir[num], i + ".txt")
                    with open(save2, 'w', encoding='utf-8') as file:
                        file.write(output_text)
                    query = 'Please output the bbox coordinates of the tampered areas in JSON format.'
                    new_message = [
                        {"role": "assistant", "content": [{"type": "text", "text": output_text}]},
                        {"role": "user", "content": [{"type": "text", "text": query}]}
                                   ]
                if num == 2:
                    save3 = os.path.join(results_dir[num], i + ".txt")
                    with open(save3, 'w', encoding='utf-8') as file:
                        file.write(output_text) 

                if output_text == 'The image is authentic and unaltered.':
                    break
                else:
                    messages.extend(new_message)

def main(val_set, gpu_select):
    num_processes = len(gpu_select)

    # TextDDLE-Test does not contain all RTM Test data. If you want to test the entire dataset, please replace it.
    image_dir = f'./datasets/TextDDLE/{val_set}/imgs'
    results_dir = [f'results/Locator/HLSCB/{val_set}/query1', f'results/Locator/HLSCB/{val_set}/query2', f'results/Locator/HLSCB/{val_set}/query3']
    for i in results_dir:
        os.makedirs(i, exist_ok=True)
    model_dir = 'saves/tvsip_locator/ep3'

    test_list = [os.path.splitext(i)[0] for i in os.listdir(image_dir)]
    test_subsets = split_list(test_list, num_processes)

    processes = []
    
    for idx, gpu_id in enumerate(gpu_select):
        p = mp.Process(target=worker, args=(gpu_id, test_subsets[idx], image_dir, results_dir, model_dir))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process images with specified validation set")
    parser.add_argument('--val_set', type=str, default='Test', 
                        help='Validation set name')
    parser.add_argument('--gpus', type=int, nargs='+', default=[5], 
                        help='GPU IDs to use (e.g., --gpus 0 1 2 3)')
    args = parser.parse_args()
    mp.set_start_method('spawn')
    main(args.val_set, args.gpus)