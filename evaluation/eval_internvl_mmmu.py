import argparse
import json
import re
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModel
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode


parser = argparse.ArgumentParser(description="MMMU Evaluation Script with InternVL")
parser.add_argument("--model_path", type=str, required=True, help="Path to the pretrained InternVL model")
parser.add_argument("--output_file", type=str, default="tmp_mmmu_evaluation_results.jsonl", help="Output JSONL file path")
args = parser.parse_args()


mmmu_cat = [
    'Accounting', 'Agriculture', 'Architecture_and_Engineering', 'Art', 'Art_Theory',
    'Basic_Medical_Science', 'Biology', 'Chemistry', 'Clinical_Medicine',
    'Computer_Science', 'Design', 'Diagnostics_and_Laboratory_Medicine', 'Economics',
    'Electronics', 'Energy_and_Power', 'Finance', 'Geography', 'History', 'Literature',
    'Manage', 'Marketing', 'Materials', 'Math', 'Mechanical_Engineering', 'Music',
    'Pharmacy', 'Physics', 'Psychology', 'Public_Health', 'Sociology'
]


all_datasets = []
for cat in mmmu_cat:
    dataset = load_dataset("path/to/MMMU/MMMU", cat, split="validation")
    all_datasets.append(dataset)
dataset_test_all = concatenate_datasets(all_datasets)
print(dataset_test_all)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModel.from_pretrained(
    args.model_path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    use_flash_attn=True,
    trust_remote_code=True
).eval().cuda()
processor = AutoTokenizer.from_pretrained(args.model_path)
tokenizer = AutoTokenizer.from_pretrained(args.model_path)


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)

    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)

    assert len(processed_images) == blocks or use_thumbnail
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image_file, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    return torch.stack(pixel_values)


OUTPUT_FILE = args.output_file
correct = 0
total = 0

with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    for example in tqdm(dataset_test_all, desc="Running Evaluation"):
        question = example["question"]
        context = example.get("context", "")
        choices = eval(example["options"])
        answer = example["answer"].strip().upper()

        pixel_values_list = []
        i = 1
        while True:
            image_key = f"image_{i}"
            if image_key in example and example[image_key] is not None:
                img = example[image_key]
                if isinstance(img, str):
                    img = Image.open(img).convert("RGB")
                single_pixel = load_image(img, max_num=1).to(torch.bfloat16).cuda()
                pixel_values_list.append(single_pixel)
                i += 1
            else:
                break

        if not pixel_values_list and example.get("image") is not None:
            img = example["image"]
            if isinstance(img, str):
                img = Image.open(img).convert("RGB")
            single_pixel = load_image(img, max_num=1).to(torch.bfloat16).cuda()
            pixel_values_list.append(single_pixel)

        if pixel_values_list:
            pixel_values = torch.cat(pixel_values_list, dim=0)
        else:
            pixel_values = torch.empty((0, 3, 224, 224), dtype=torch.bfloat16, device="cuda")

        choice_text = "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(choices)])
        prompt = (
            f"<image>\n{question}\n{choice_text}\n"
            "Solve the problem through step-by-step reasoning and answer directly with the option letter. "
            "Think about the reasoning process first and answer the question following this format: "
            "<think> THINKING </think><answer> ANSWER </answer>."
        )

        generation_config = dict(max_new_tokens=1024, do_sample=False)
        response = model.chat(tokenizer, pixel_values, prompt, generation_config)

        pred_answer = None
        is_correct = False
        extraction_method = "failed"

        match = re.search(r"<answer>\s*([A-Z])\s*</answer>", response, re.IGNORECASE)
        if match:
            pred_answer = match.group(1).strip().upper()
            extraction_method = "regex"
            is_correct = (pred_answer == answer)
            if is_correct:
                correct += 1
        else:
            fallback_matches = re.findall(r"\b([A-Z])\b", response.upper())
            if fallback_matches:
                pred_answer = fallback_matches[-1]
                extraction_method = "fallback"
                print(f"[WARN] Using fallback answer: {pred_answer} for response: {response}")
                is_correct = (pred_answer == answer)
                if is_correct:
                    correct += 1
            else:
                print(f"[ERROR] No valid answer found in response: {response}")

        result_data = {
            "question_id": total,
            "question": question,
            "context": context,
            "choices": choices,
            "correct_answer": answer,
            "predicted_answer": pred_answer,
            "is_correct": is_correct,
            "response": response,
            "extraction_method": extraction_method,
            "image_count": len(pixel_values_list)
        }

        f.write(json.dumps(result_data, ensure_ascii=False) + "\n")
        f.flush()
        total += 1

accuracy = correct / total if total > 0 else 0
print(f"\n✅ Accuracy on test split: {accuracy*100:.2f}%")
