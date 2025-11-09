import argparse
import json
import re
import io
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset, concatenate_datasets
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info 
from torch.utils.data import DataLoader




parser = argparse.ArgumentParser(description="Evaluate Qwen2.5-VL model on MMMU dataset.")
parser.add_argument("--model_path", type=str, required=True, help="Path to the pretrained Qwen2.5-VL model")
parser.add_argument("--dataset_path", type=str, default="MMMU/MMMU", help="Path or name of the MMMU dataset on Hugging Face Hub")
parser.add_argument("--output_file", type=str, default="mmmu_evaluation_results.jsonl", help="Output JSONL file path")
parser.add_argument("--batch_size", type=int, default=4, help="Batch size for evaluation")
parser.add_argument("--prompt" , type=str, default="Solve the problem through step-by-step reasoning and answer directly with the option letter. Think about the reasoning process first and answer the question following this format: <think> THINKING </think><answer> ANSWER </answer>.", help="Prompt template to use")
args = parser.add_args()




mmmu_cat = [
    'Accounting', 'Agriculture', 'Architecture_and_Engineering', 'Art', 'Art_Theory',
    'Basic_Medical_Science', 'Biology', 'Chemistry', 'Clinical_Medicine',
    'Computer_Science', 'Design', 'Diagnostics_and_Laboratory_Medicine', 'Economics',
    'Electronics', 'Energy_and_Power', 'Finance', 'Geography', 'History', 'Literature',
    'Manage', 'Marketing', 'Materials', 'Math', 'Mechanical_Engineering', 'Music',
    'Pharmacy', 'Physics', 'Psychology', 'Public_Health', 'Sociology'
]

print("Loading MMMU dataset...")
all_datasets = []
for cat in tqdm(mmmu_cat, desc="Loading categories"):
    try:
        
        dataset = load_dataset(args.dataset_path, cat, split="validation")
    except ValueError:
        try:
            print(f"Warning: 'validation' split not found for {cat}. Trying 'test' split.")
            dataset = load_dataset(args.dataset_path, cat, split="test")
        except Exception as e:
            print(f"Error loading category {cat}: {e}. Skipping.")
            continue
    all_datasets.append(dataset)

if not all_datasets:
    print("Error: No datasets were loaded. Check --dataset_path and dataset availability.")
    exit(1)

dataset_test_all = concatenate_datasets(all_datasets)
print(f"Loaded {len(dataset_test_all)} samples from {len(all_datasets)} categories.")





print("Loading Qwen model and processor...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    args.model_path, torch_dtype="auto", device_map="auto"
)
processor = AutoProcessor.from_pretrained(args.model_path)
print("Model loaded successfully.")




def collate_fn(batch):
    """Custom collate function for batching MMMU data"""
    questions = []
    choices_list = []
    images_list = [] 
    answers = []
    
    for example in batch:
        questions.append(example["question"])
        
        
        try:
            choices = eval(example["options"]) if isinstance(example["options"], str) else example["options"]
        except Exception:
            choices = example["options"] 
        choices_list.append(choices)

        answers.append(example["answer"].strip().upper())
        
        example_images = []
        
        i = 1
        while True:
            image_key = f"image_{i}"
            if image_key in example and example[image_key] is not None:
                img = example[image_key]
                if isinstance(img, str):
                    img = Image.open(img).convert("RGB")
                elif isinstance(img, dict) and 'bytes' in img:
                     img = Image.open(io.BytesIO(img['bytes'])).convert("RGB")
                elif isinstance(img, Image.Image):
                     img = img.convert("RGB") 
                
                example_images.append(img)
                i += 1
            else:
                break
        
        
        if not example_images and example.get("image") is not None:
            img = example["image"]
            if isinstance(img, str):
                img = Image.open(img).convert("RGB")
            elif isinstance(img, dict) and 'bytes' in img:
                 img = Image.open(io.BytesIO(img['bytes'])).convert("RGB")
            elif isinstance(img, Image.Image):
                 img = img.convert("RGB")
            example_images.append(img)

        
        images_list.append(example_images if example_images else [None])
    
    return {
        "questions": questions,
        "choices_list": choices_list,
        "images_list": images_list, 
        "answers": answers
    }




dataloader = DataLoader(
    dataset_test_all,
    batch_size=args.batch_size,
    shuffle=False,
    collate_fn=collate_fn
)




correct = 0
total = 0
OUTPUT_FILE = args.output_file


generation_config = dict(
    max_new_tokens=4096,
    do_sample=False
)

with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Running Batch Evaluation")):
        questions = batch["questions"]
        choices_list = batch["choices_list"]
        images_list = batch["images_list"] 
        answers = batch["answers"]
        
        batch_size = len(questions)

        messages = []
        result_data_list = []
        for i in range(batch_size):
            choice_text = "\n".join([f"{chr(65+idx)}. {opt}" for idx, opt in enumerate(choices_list[i])])
            formatted_question = (
                f"{questions[i]}\n{choice_text}\n"
                f"{args.prompt}"
            )
            
            message = [
                {
                    "role": "user",
                    "content": [],
                }
            ]
            
            
            sample_images = images_list[i]
            image_count = 0
            for image in sample_images:
                if image is not None:
                    message[0]["content"].append({"type": "image", "image": image})
                    image_count += 1
            
            
            message[0]["content"].append({"type": "text", "text": f"{formatted_question}"})
            messages.append(message)
            
            
            result_data_list.append({
                "index": batch_idx * args.batch_size + i,
                "answer": answers[i],
                "formatted_question": formatted_question,
                "question": questions[i],
                "choices": choices_list[i],
                "image_count": image_count
            })
        
        
        texts = [
            processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            for msg in messages
        ]
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            padding_side="left",
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")
        
        generated_ids = model.generate(**inputs, **generation_config)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_texts = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        

        for i, output_text in enumerate(output_texts):
            result_data = result_data_list[i]
            result_data["gen"] = output_text

            
            
            final_answer_match = re.search(r"\\boxed\{([A-Z])\}", output_text)
            if final_answer_match:
                predicted_answer = final_answer_match.group(1)
            else:
                
                
                cleaned_output = re.sub(r"<think>.*?</think>", "", output_text, flags=re.DOTALL)
                fallback_matches = re.findall(r"\b([A-Z])\b", cleaned_output)
                if fallback_matches:
                    predicted_answer = fallback_matches[-1]
                    
                else:
                    predicted_answer = "N/A"
                    

            result_data["pred"] = predicted_answer

            
            if predicted_answer == result_data["answer"]:
                correct += 1
                result_data["correct"] = True
            else:
                result_data["correct"] = False
            
            total += 1 

        
        for result in result_data_list:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
            f.flush()




accuracy = correct / total if total > 0 else 0
print(f"\n✅ Accuracy on MMMU test split: {accuracy*100:.2f}%")
print(f"📊 Total samples processed: {total}")
print(f"📁 Results saved to: {OUTPUT_FILE}")