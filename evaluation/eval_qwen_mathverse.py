import argparse
import json
import re
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset
from torch.utils.data import DataLoader



from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info 




parser = argparse.ArgumentParser(description="Evaluate Qwen2.5-VL model on MathVerse testmini set.")
parser.add_argument("--model_path", type=str, required=True, help="Path to the pretrained Qwen-VL model")
parser.add_argument("--output_file", type=str, default="evaluation_results.jsonl", help="Output JSONL file path")
parser.add_argument("--batch_size", type=int, default=4, help="Batch size for evaluation")

parser.add_argument("--prompt" , type=str, default="Solve the problem through step-by-step reasoning and answer directly with the option letter. Think about the reasoning process first and answer the question following this format: <think> THINKING </think><answer> ANSWER </answer>.", help="Prompt template to use")
args = parser.parse_args()





dataset_test = load_dataset("path/to/MathVerse", "testmini", split="testmini")
print(dataset_test)




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    args.model_path,
    torch_dtype="auto", 
    device_map="auto"   
).eval()
processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)















def collate_fn(batch):
    questions, contexts, images, answers = [], [], [], []
    for example in batch:
        questions.append(example["question"])
        contexts.append(example.get("context", ""))
        image = example["image"]
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        
        if image is None:
            images.append(None)
        else:
            images.append(image)
        answers.append(example["answer"].strip().upper())
    return {
        "questions": questions,
        "contexts": contexts,
        "images": images,
        "answers": answers
    }




dataloader = DataLoader(
    dataset_test,
    batch_size=args.batch_size,
    shuffle=False,
    collate_fn=collate_fn,
    num_workers=4 
)




correct = 0
total = 0

generation_config = dict(
    max_new_tokens=1024, 
    do_sample=False
)

with open(args.output_file, 'w', encoding='utf-8') as f:
    for batch in tqdm(dataloader, desc="Running Batch Evaluation"):
        questions = batch["questions"]
        contexts = batch["contexts"]
        images = batch["images"]
        answers = batch["answers"]
        batch_size = len(questions)

        
        
        
        messages = []
        result_data_list = []
        for i in range(batch_size):
            
            
            formatted_question = f"{questions[i]}\n{args.prompt}"
            
            message = [
                {
                    "role": "user",
                    "content": [],
                }
            ]
            
            if images[i] is not None:
                message[0]["content"].append({"type": "image", "image": images[i]})
            
            
            message[0]["content"].append({"type": "text", "text": f"{formatted_question}"})
            messages.append(message)
            
            
            result_data_list.append({
                "question_id": total + i,
                "question": questions[i],
                "context": contexts[i],
                "correct_answer": answers[i],
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
        inputs = inputs.to(device)

        
        generated_ids = model.generate(**inputs, **generation_config)
        
        
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_texts = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        
        
        
        for i, response in enumerate(output_texts):
            result_data = result_data_list[i]
            answer = result_data["correct_answer"]
            result_data["response"] = response

            
            match = re.search(r"\\boxed\{([A-Z])\}", response, re.IGNORECASE)
            
            pred_answer = None
            is_correct = False
            extraction_method = "regex_boxed"

            if match:
                pred_answer = match.group(1).strip().upper()
                is_correct = (pred_answer == answer)
            else:
                
                fallback_matches = re.findall(r"\b([A-Z])\b", response.upper())
                if fallback_matches:
                    pred_answer = fallback_matches[-1]
                    extraction_method = "fallback"
                    is_correct = (pred_answer == answer)
                    print(f"[WARN] Using fallback answer: {pred_answer} for response: {response}")
                else:
                    extraction_method = "failed"
                    pred_answer = "N/A" 
                    print(f"[ERROR] No valid answer found in response: {response}")

            if is_correct:
                correct += 1

            result_data["predicted_answer"] = pred_answer
            result_data["is_correct"] = is_correct
            result_data["extraction_method"] = extraction_method
            
            
            f.write(json.dumps(result_data, ensure_ascii=False) + '\n')
            f.flush()
            total += 1




accuracy = correct / total if total > 0 else 0
print(f"\n✅ Accuracy on test split: {accuracy*100:.2f}%")
print(f"📊 Total samples processed: {total}")
print(f"📁 Results saved to: {args.output_file}")