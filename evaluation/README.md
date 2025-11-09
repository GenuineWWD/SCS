# EVALUATION PART OF Self-Consistency Sampling

This folder contains the evaluation code for the Self-Consistency Sampling (SCS) method introduced in our NeurIPS 2025 paper. The evaluation scripts are designed to assess the performance of open source model Qwen/InternVL trained with SCS on various multimodal benchmarks.

## Evaluation Benchmarks
- M3CoT
- MathVerse
- MathVision
- MMMU
- ScienceQA
- WeMath

## Setup Instructions

1. **Install Dependencies**:
   Ensure you have Python 3.8+ installed. Install the required packages using pip:
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare Datasets**:
    Download and preprocess the evaluation datasets. Ensure they are formatted correctly for the evaluation scripts.
    - M3CoT dataset can be found at: [M3CoT Dataset](https://huggingface.co/datasets/LightChen2333/M3CoT)
    - MathVerse dataset can be found at: [MathVerse Dataset](https://huggingface.co/datasets/AI4Math/MathVerse)
    - MathVision dataset can be found at: [MathVision Dataset](https://huggingface.co/datasets/MathLLMs/MathVision)
    - MMMU dataset can be found at: [MMMU Dataset](https://huggingface.co/datasets/MMMU/MMMU)
    - ScienceQA dataset can be found at: [ScienceQA Dataset](https://huggingface.co/datasets/derek-thomas/ScienceQA)
    - WeMath dataset can be found at: [WeMath Dataset](http://huggingface.co/datasets/We-Math/We-Math)
3. **Run Evaluation**:
    Use the provided evaluation scripts to assess model (QwenVL/InternVL) performance on each benchmark. For example, to evaluate on M3CoT, run:
    ```bash
    cd evaluation
    python eval_internvl_m3cot.py --model_path /path/to/your/model \
    --model_path /path/to/your/model --output_path /path/to/save/results.jsonl
    ```
    
    Duration of some evaluations may vary based on dataset size. We provide ```--batch_size``` argument to control the evaluation speed and memory usage for m3cot, mathverse, mathvision, scienceqa, and wemath evaluations.
4. **NOTE**: 

- For original InternVL or QwenVL model evaluation, please use CoT prompt provided by [MMMU Prompts](https://github.com/MMMU-Benchmark/MMMU/blob/main/mmmu-pro/prompts.yaml)
- For Trained with SCS model evaluation, please use CoT prompt provided by SCS Prompts:
    > Solve the problem through step-by-step reasoning and answer directly with the option letter. Think about the reasoning process first and answer the question following this format: <think> THINKING </think><answer> ANSWER </answer>.
