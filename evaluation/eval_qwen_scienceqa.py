import pandas as pd
from pathlib import Path
from datasets import load_dataset
import argparse
import torch
from tqdm import tqdm
import re 
import json
from PIL import Image 
import torch 
from torch.utils.data import DataLoader
import base64
import io


from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor

from qwen_vl_utils import process_vision_info




parser = argparse.ArgumentParser(description="Evaluate Qwen model on ScienceQA dataset.")
parser.add_argument("--model_path", type=str, required=True, help="Path to the pretrained Qwen model")
parser.add_argument("--output_file", type=str, default="evaluation_results.jsonl", help="Output JSONL file path")
parser.add_argument("--batch_size", type=int, default=4, help="Batch size for evaluation")

parser.add_argument("--prompt" , type=str, default="Solve the problem through step-by-step reasoning and answer directly with the option letter. Think about the reasoning process first and answer the question following this format: <think> THINKING </think><answer> ANSWER </answer>.", help="Prompt template to use")
args = parser.parse_args()



dataset_test = load_dataset("path/to/ScienceQA/data", split="test")

print(dataset_test)


import torch
from tqdm import tqdm
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    args.model_path, torch_dtype="auto", device_map="auto"
)
processor = AutoProcessor.from_pretrained(args.model_path)










def collate_fn(batch):
    """Custom collate function for batching"""
    questions = []
    contexts = []
    choices_list = []
    images = []
    answers = []
    
    for example in batch:
        questions.append(example["lecture"] + example["question"])
        contexts.append(example.get("context", ""))
        choices_list.append(example["choices"])
        
        
        if example["image"] is not None:
            image_data = example["image"]['bytes']

            
            image = Image.open(io.BytesIO(image_data))
            images.append(image)
        else:
            images.append(None)
        
        answers.append(['A','B','C','D','E','F','G','H','I'][example["answer"]])
    
    return {
        "questions": questions,
        "contexts": contexts, 
        "choices_list": choices_list,
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
    max_new_tokens=4096,
    do_sample=False
)




with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Running Batch Evaluation")):
        questions = batch["questions"]
        contexts = batch["contexts"]
        choices_list = batch["choices_list"]
        images = batch["images"]
        answers = batch["answers"]
        
        batch_size = len(questions)

        messages = []
        result_data_list = []
        for i in range(batch_size):
            choice_text = "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(choices_list[i])])
            formatted_question = (
                f"{questions[i]}\n{choice_text}\n"
                f"{args.prompt}"
            )
            message = [
                {
                    "role": "user",
                    "content": [
                    ],
                }
            ]
            if images[i] is not None:
                if isinstance(images[i], list):
                    for image in images[i]:
                        message[0]["content"].append({"type": "image", "image": image})
                else:
                    message[0]["content"].append({"type": "image", "image": images[i]})
            message[0]["content"].append({"type": "text", "text": f"{formatted_question}"})
            messages.append(message)
            result_data_list.append({
                "index": batch_idx * BATCH_SIZE + i,
                "answer": answers[i],
                "formatted_question": formatted_question,
                "question": questions[i],
                "context": contexts[i],
                "choices": choices_list[i],
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

            
            final_answer_match = re.search(r"\\boxed\{([A-I])\}", output_text)
            if final_answer_match:
                predicted_answer = final_answer_match.group(1)
            else:
                predicted_answer = "N/A"
            result_data["pred"] = predicted_answer

            
            if predicted_answer == result_data["answer"]:
                correct += 1
                result_data["correct"] = True
            else:
                result_data["correct"] = False

        
        for i, result in enumerate(result_data_list):
            
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
            f.flush()  
            
            total += 1


accuracy = correct / total if total > 0 else 0
print(f"\n✅ Accuracy on test split: {accuracy*100:.2f}%")
print(f"📊 Total samples processed: {total}")
print(f"📁 Results saved to: {OUTPUT_FILE}")