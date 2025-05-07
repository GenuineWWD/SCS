import json
import os
import random
import re
from argparse import ArgumentParser
from multiprocessing import Process, Queue

import Levenshtein
from flask import Flask, jsonify, request


app = Flask(__name__)

problem_to_answer = {}
OPTION_LIST = ['A','B','C','D','E','F','G','H','I','J','K']

def get_response_from_query(q: str):
    ends_of_sentence = ["<|im_end|>", "<｜end▁of▁sentence｜>", "<|endoftext|>"]
    pos = re.search(response_prefix, q)
    if pos is None:
        return q
    response = q[pos.end() :]
    for e in ends_of_sentence:
        response = response.replace(e, "")
    return response.strip()


def verify_format(content):
    
    content = extract_answer(content).strip()

    return content in OPTION_LIST

def find_similar_problem(problem):
    max_sim = -1
    target_problem = None
    for p in problem_to_answer.keys():
        sim = Levenshtein.ratio(problem, p)
        if sim > max_sim:
            max_sim = sim
            target_problem = p
    return target_problem

def extract_answer(ans):
    start_tag = r'\boxed{'

    # 确保只存在一个匹配项
    if ans.count(start_tag) != 1:
        return ans

    idx = ans.find(start_tag)
    idx += len(start_tag)

    brace_level = 1
    content_start = idx
    i = idx

    while i < len(ans):
        if ans[i] == '{':
            brace_level += 1
        elif ans[i] == '}':
            brace_level -= 1
            if brace_level == 0:
                # 找到完整闭合的花括号
                return ans[content_start:i]
        i += 1

    # 若未找到匹配的闭合括号
    return ans


def consistency_reward(responses):
    answer_set = set()
    for key,value in responses.items():
        parse_answer = extract_answer(value).strip()

        answer_set.add(parse_answer)

    return (5 - len(answer_set))*0.3

def verify_option(input_queue, output_queue):
    while True:
        content, sol = input_queue.get()
        if len(sol) != 0:
            # We require the answer to be provided in correct latex (no malformed operators)
            # Reward 1 if the content is the same as the ground truth, 0 otherwise
           parse_answer = extract_answer(content).strip()
           
           reward=1.0 if parse_answer==sol[0] else 0.0
        else:
            # If the gold solution is not parseable, we reward 1 to skip this example
            reward = 1.0
            print("Failed to parse gold solution: ", sol)

        output_queue.put(reward)


@app.route("/get_reward", methods=["POST"])
def get_reward():
    # 获取请求中的 JSON 数据
    data = request.get_json()
    # 检查是否有 'query' 字段
    if "query" not in data:
        return jsonify({"error": "queries field is required"}), 400
    #print(data['cut_response_text_dict'])
    rewards = []
    for q,problem in zip(data["query"],data["prompts"]):
        if problem is None:
            return jsonify({"error": f"problem not found from {q}"}), 400
        if problem not in problem_to_answer:
            # This should not happen
            print(f"problem not exists: {problem}")
            problem = find_similar_problem(problem)
        answer = problem_to_answer[problem]
        #response = remove_short_string_part(q,problem)
        response = get_response_from_query(q) or q
        if response is None:
            return jsonify({"error": f"response not found from {q}"}), 400
        format_reward = float(verify_format(response))
        input_queue.put((response, answer))
        acc_reward = float(output_queue.get())
        consist_r = consistency_reward(data['cut_response_text_dict'])
        do_print = random.randint(1, 20) == 1
        #do_print = False
        if do_print:
            info=f"Query: {q}\n\nProblem: {problem}\n\n Answer: {answer}\n\n Response: {response}\n\n Format Reward: {format_reward}\n\n Acc Reward: {acc_reward}\n\nConsistency Reward: {consist_r}\n\n"
            info = re.sub(r"<\|.*?\|>","",info)
            print(info)
            
        rewards.append(0.5 * format_reward + acc_reward + consist_r)
    # 返回包含 rewards 的响应
    return jsonify({"rewards": rewards})


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, default=None, help="Datasets to use (comma separated)", required=True
    )
    parser.add_argument(
        "--prompt-template", type=str, default=None, help="Prompt template", required=True
    )
    parser.add_argument(
        "--input_key", type=str, default="prompt", help="The key name of prompt."
    )
    args = parser.parse_args()
    
    # Split dataset paths and load all datasets
    dataset = []
    for dataset_path in args.dataset.split(','):
        dataset_path = dataset_path.strip()
        if dataset_path.endswith("json"):
            with open(dataset_path, "r") as f:
                dataset.extend(json.load(f))
        elif dataset_path.endswith("jsonl"):
            with open(dataset_path, "r") as f:
                dataset.extend([json.loads(l) for l in f.readlines()])
        else:
            raise ValueError(f"Unsupported file format for dataset: {dataset_path}")

    #format_pattern = r"^<think>(?:(?!</think>).)*</think><answer>(?:(?!</answer>).)*</answer>\Z"
    format_pattern =  r"^<think>(?:(?!</think>).)*</think>.*<answer>(?:(?!</answer>).)*</answer>\Z"

    if args.prompt_template=="chatml":
        problem_pattern = r"<\|im_start\|>user\n(.*?)<\|im_end\|>"
        response_prefix = r"<\|im_start\|>assistant\n"
    elif args.prompt_template=="qwen1":
        problem_pattern = r"｜User｜>(.*?)<｜Assistant｜>"
        response_prefix = r"<｜Assistant｜>"
    elif args.prompt_template=="base":
        problem_pattern = r"User: (.*?)\n\nAssistant:"
        response_prefix = r"Assistant: "
    else:
        raise ValueError(f"Unknown chat format: {args.dataset}")
    print("load dataset success")
    for item in dataset:
        problem = item[args.input_key]
        #problem = item["question"]
        answer = item["answer"].strip()
        # we require the answer to be in latex format
        problem_to_answer[problem] = answer

    # math_verify can only run in main thread
    input_queue = Queue()
    output_queue = Queue()
    p = Process(target=verify_option, args=(input_queue, output_queue))
    p.start()

    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)
    p.kill()
