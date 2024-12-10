# QA Prompting: Improving Summarization using Question-Answering in Large Language Models

---

## Overview

This repository contains the codebase for generating summaries and performing analyses using question-answering (QA) prompting techniques. It supports multiple tasks such as dataset analysis, QA relevance evaluation, summary generation, and summarization analysis. The project leverages HuggingFace models and provides tools to streamline the evaluation and experimentation processes.

---

## Table of Contents

- [Repository Installation](#repository-installation)
- [Usage](#usage)
- [How to Run](#how-to-run)
- [File Structure](#file-structure)

---


## Repository Installation

### Step 1: Clone the Repository
```bash
git clone https://github.com/neelabhsinha/qa-prompting.git
cd qa-prompting
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Configure API Keys
Set your HuggingFace API key to access pre-trained models:
```bash
export HF_API_KEY='your_huggingface_api_key_here'
```

---

## Usage

### Available Tasks

The following tasks are supported:
1. `analyze_dataset`: Analyze the dataset for task-specific insights.
2. `qa_generate`: Generate answers for QA tasks.
3. `analyze_qa_relevance`: Analyze the relevance of QA tasks.
4. `summary_generate`: Generate summaries using QA-based approaches.
5. `analyze_summarization`: Analyze the quality of generated summaries.

### Supported Models

This project supports models from HuggingFace's library. The specific model can be set using the HuggingFace ID via `--model_name` argument.

### Metrics

The repository supports the following evaluation metrics:
- ROUGE (`rouge1`, `rouge2`, `rougeL`)
Future additions -
- METEOR
- BERTScore (`precision`, `recall`, `f1`)

### Configurations

- **Batch Size**: Specify the batch size for processing tasks (`--batch_size`).
- **Data Split**: Choose between `train` and `test` splits (`--split`).
- **Evaluation Metric**: Set the metric for evaluation (`--metric`).

---

## How to Run

### Analyze Dataset
```bash
python main.py --task analyze_dataset --split test --instance_per_task 100
```

### Generate QA Answers
```bash
python main.py --task qa_generate --model_name meta-llama/Llama-3.2-3B --batch_size 4 --instance_per_task 100
```

### Generate Summaries
```bash
python main.py --task summary_generate --model_name meta-llama/Llama-3.2-3B --batch_size 4 --instance_per_task 100 --top_k 5
```

### Analyze Summarization
```bash
python main.py --task analyze_summarization --metric rougeL
```

---

## File Structure

### Key Files and Directories
- **`main.py`**: The main entry point for executing tasks.
- **`const.py`**: Contains task-specific constants and configurations.
- **`src/`**:
  - **`analysis/`**:
    - `qa_relevance_analyzer.py`: Analyze QA relevance across tasks.
    - `summarization_analyzer.py`: Analyze summarization results using specified metrics.
    - `super_natural_instructions_analyzer.py`: Analyze dataset-specific metrics and relevance.
  - **`handler/`**:
    - `exit_handler.py`: Handle exit signals gracefully.
  - **`loader/`**:
    - `super_natural_instructions.py`: Load task instances from the dataset.
    - `super_natural_instructions_loader.py`: Handle batching and post-processing for loaded data.
  - **`metrics/`**:
    - `bert_score.py`: Calculate BERTScore for predictions.
    - `bleu_score.py`: Compute BLEU scores for generated summaries.
    - `meteor.py`: Compute METEOR scores for generated summaries.
    - `qa_precision_score.py`: Evaluate QA precision scores for generated answers.
    - `rouge.py`: Calculate ROUGE metrics for generated summaries.
  - **`models/`**:
    - `gemini.py`: Interface for running the Gemini model.
    - `gpt.py`: Interface for GPT-based models.
    - `model.py`: Generic model utility functions and base classes.
  - **`prompts/`**:
    - `qa_prompts.py`: Define QA-specific prompts for input preparation.
    - `summarization_prompt.py`: Define prompts for summarization tasks.
    - `super_natural_instructions_prompt.py`: Generate prompts for tasks based on SuperNatural Instructions.
  - **`utils/`**:
    - `model_pass.py`: Facilitate model inference with multiple configurations.
    - `qa_generation.py`: Generate QA-based answers from input data.
    - `results_io_util.py`: Read and write utilities for managing evaluation results.
    - `summary_generation.py`: Generate summaries using the QA-based approach.

---