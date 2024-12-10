import argparse
import os

from const import tasks
from src.analysis.qa_relevance_analyzer import QARelevanceAnalyzer
from src.analysis.summarization_analyzer import SummarizationAnalyzer
from src.analysis.super_natural_instructions_analyzer import SuperNaturalInstructionsAnalyzer

from src.utils.qa_generation import generate_answers
from src.utils.summary_generation import generate_summary


def configure_huggingface():
    try:
        hf_token = os.getenv('HF_API_KEY')  # Make sure to add HF_API_KEY to environment variables
        # Add it in .bashrc or .zshrc file to access it globally
        os.environ['HF_TOKEN'] = hf_token
    except (TypeError, KeyError):
        print('Not able to set HF token. Please set HF_API_KEY in environment variables.')


def get_args():
    parser = argparse.ArgumentParser(
        description='Project for generating summaries using QA prompting.')

    # General execution parameters
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Define the batch size for model training or evaluation. Default is 1.")

    # Task-related parameters
    parser.add_argument("--task", type=str, default='eval', choices=tasks,
                        help="Specify the task to perform. Options are based on predefined tasks in the 'tasks'"
                             " module.")
    parser.add_argument("--model_name", type=str, default=None,
                        help="Model to be used for the evaluation or any other specified task.")
    parser.add_argument("--split", type=str, default='train', choices=['train', 'test'],
                        help="Define the data split to be used. Options are 'train' or 'test'. Default is 'test'.")

    parser.add_argument('--top_k', type=int, default=5,
                        help='Set the number of top questions to use for generating summaries. Default is 5.')

    # Filtering and selection options (filtering entities are taken from const.py file)
    parser.add_argument('--instance_per_task', type=int, default=50000,
                        help='Set the maximum number of instances per task to process. Default is 50000.')

    # Evaluation configuration
    parser.add_argument('--metric', default='rougeL', type=str,
                        help='Specify the evaluation metric to use. Default is "rougeL". Options might include'
                             ' various NLP-specific metrics like BLEU, METEOR, etc., depending on what is implemented.')

    # Checkpoint handling (if evaluating from a local checkpoint)
    parser.add_argument('--checkpoint', type=str, default='none',
                        help='Specify the checkpoint folder name if resuming from a saved state. Use "none"'
                             ' to start from scratch.')
    
    parser.add_argument('--baseline', action='store_true', default=False,
                        help='Use this flag to generate summaries without ICL examples and questions.')

    return parser.parse_args()


if __name__ == '__main__':
    configure_huggingface()
    args = get_args()
    ckp = None if args.checkpoint == 'none' else args.checkpoint
    if args.task == 'analyze_dataset':
        analyzer = SuperNaturalInstructionsAnalyzer(split=args.split, instance_per_task=args.instance_per_task)
        analyzer.save_analysis_results()
    if args.task == 'qa_generate':
        generate_answers(args.model_name, args.batch_size, args.instance_per_task, ckp)
    if args.task == 'analyze_qa_relevance':
        analyzer = QARelevanceAnalyzer()
        analyzer.collect_all_results()
    if args.task == 'summary_generate':
        generate_summary(args.model_name, args.batch_size, args.instance_per_task, args.top_k, args.baseline, ckp)
    if args.task == 'analyze_summarization':
        analyzer = SummarizationAnalyzer(metric=args.metric)
        analyzer.save_results()
