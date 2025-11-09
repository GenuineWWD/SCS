import pandas as pd
from pathlib import Path
from datasets import load_dataset
import argparse

dataset_test = load_dataset("path/to/We-Math",split = "testmini")

print(dataset_test)

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




parser = argparse.ArgumentParser(description="Evaluate InternVL model on We-Math dataset")
parser.add_argument("--model_path", type=str, required=True, help="Path to the pretrained model directory")
parser.add_argument("--output_file", type=str, default="evaluation_results.jsonl", help="Path to save the evaluation results")
parser.add_argument("--batch_size", type=int, default=4, help="Batch size for evaluation")
args = parser.parse_args()


path = args.model_path
model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    use_flash_attn=True,
    trust_remote_code=True).eval().cuda()
processor = AutoTokenizer.from_pretrained(path)
tokenizer = AutoTokenizer.from_pretrained(path)

import torchvision.transforms as T

from torchvision.transforms.functional import InterpolationMode

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform


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
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    
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
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image_file, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


import re 
import json
from tqdm import tqdm 
from PIL import Image 
import torch 
from torch.utils.data import DataLoader

def collate_fn(batch):
    """Custom collate function for batching"""
    questions = []
    contexts = []
    images = []
    choices_texts = []
    answers = []
    
    for example in batch:
        questions.append(example["question"])
        contexts.append(example.get("context", ""))
        choices_texts.append(example.get("option", ""))
        
        
        image = example["image_path"]
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        images.append(image)
        
        answers.append(example["answer"].strip().upper())
    
    return {
        "questions": questions,
        "contexts": contexts, 
        "choices_text": choices_texts,
        "images": images,
        "answers": answers
    }


BATCH_SIZE = args.batch_size  
OUTPUT_FILE = args.output_file 


dataloader = DataLoader(
    dataset_test, 
    batch_size=BATCH_SIZE, 
    shuffle=False, 
    collate_fn=collate_fn
)

correct = 0
total = 0


generation_config = dict(
    max_new_tokens=1024,
    do_sample=False
)


with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    for batch in tqdm(dataloader, desc="Running Batch Evaluation"):
        questions = batch["questions"]
        contexts = batch["contexts"]
        choices_texts = batch["choices_text"]
        images = batch["images"]
        answers = batch["answers"]
        
        batch_size = len(questions)
        
        
        pixel_values_list = []
        image_counts = []
        
        for image in images:
            pixel_values = load_image(image, max_num=12).to(torch.bfloat16).cuda()
            pixel_values_list.append(pixel_values)
            image_counts.append(pixel_values.size(0))  
        
        
        pixel_values_batch = torch.cat(pixel_values_list, dim=0)
        
        
        formatted_questions = []
        for i in range(batch_size):
            choice_text = choices_texts[i]
            formatted_question = (
                f"{questions[i]}\n{choice_text}\n"
                "Solve the problem through step-by-step reasoning and answer directly with the option letter. "
                "Think about the reasoning process first and answer the question following this format: "
                "<think> THINKING </think><answer> ANSWER </answer>."
            )
            formatted_questions.append(formatted_question)
        
        
        try:
            responses = model.batch_chat(
                tokenizer, 
                pixel_values_batch,
                image_counts=image_counts,
                questions=formatted_questions,
                generation_config=generation_config
            )
        except Exception as e:
            print(f"[ERROR] Batch chat failed: {e}")
            
            responses = []
            for i in range(batch_size):
                try:
                    pixel_values = pixel_values_list[i]
                    response = model.chat(
                        tokenizer,
                        pixel_values,
                        formatted_questions[i],
                        generation_config
                    )
                    responses.append(response)
                except Exception as individual_e:
                    print(f"[ERROR] Individual chat failed for sample {i}: {individual_e}")
                    responses.append("")
        
        
        for i, response in enumerate(responses):
            answer = answers[i]
            question = questions[i]
            context = contexts[i]
            
            
            match = re.search(r"<answer>\s*([A-Z])\s*</answer>", response, re.IGNORECASE)
            pred_answer = None
            is_correct = False
            extraction_method = "regex"
            
            if match:
                pred_answer = match.group(1).strip().upper()
                is_correct = (pred_answer == answer)
                if is_correct:
                    correct += 1
            else:
                
                fallback_matches = re.findall(r"\b([A-Z])\b", response.upper())
                if fallback_matches:
                    pred_answer = fallback_matches[-1]
                    extraction_method = "fallback"
                    is_correct = (pred_answer == answer)
                    print(f"[WARN] Using fallback answer: {pred_answer} for response: {response}")
                    if is_correct:
                        correct += 1
                else:
                    pred_answer = None
                    extraction_method = "failed"
                    print(f"[ERROR] No valid answer found in response: {response}")
            
            
            result_data = {
                "question_id": total,  
                "question": question,
                "context": context,
                "correct_answer": answer,
                "predicted_answer": pred_answer,
                "is_correct": is_correct,
                "response": response,
                "extraction_method": extraction_method,
                "image_patch_count": image_counts[i]  
            }
            
            
            f.write(json.dumps(result_data, ensure_ascii=False) + '\n')
            f.flush()  
            
            total += 1


accuracy = correct / total if total > 0 else 0
print(f"\n✅ Accuracy on test split: {accuracy*100:.2f}%")
print(f"📊 Total samples processed: {total}")
print(f"📁 Results saved to: {OUTPUT_FILE}")