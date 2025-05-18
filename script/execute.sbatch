#!/bin/bash
#SBATCH -J Generate
#SBATCH -N1 --ntasks-per-node=1
#SBATCH --gres=gpu:H200:1
#SBATCH --mem-per-gpu=80GB
#SBATCH -t5:00:00
#SBATCH -ologs/Analysis-%j.out

export PYTHONUNBUFFERED=TRUE
source ~/.bashrc
conda activate nlp_env
cd ~/scratch/summarization

# Default arguments
batch_size=1
task="eval"
model_name="meta-llama/Meta-Llama-3-8B-Instruct"
split="train"
top_k=None
instance_per_task=50000
metric="bert_score_recall"
checkpoint="none"
baseline_flag=""

# Parse command-line arguments
while [ "$#" -gt 0 ]; do
  case "$1" in
    --batch_size) batch_size="$2"; shift 2 ;;
    --task) task="$2"; shift 2 ;;
    --model_name) model_name="$2"; shift 2 ;;
    --split) split="$2"; shift 2 ;;
    --top_k) top_k="$2"; shift 2 ;;
    --instance_per_task) instance_per_task="$2"; shift 2 ;;
    --metric) metric="$2"; shift 2 ;;
    --checkpoint) checkpoint="$2"; shift 2 ;;
    --baseline) baseline_flag="--baseline"; shift 1 ;; # Add the baseline flag
    --global_top_k) global_top_k="--global_top_k"; shift 1 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

# Construct the command dynamically
cmd="srun -u python -u main.py --batch_size $batch_size --task $task --model_name $model_name --split $split --instance_per_task $instance_per_task --metric $metric --checkpoint $checkpoint"

# Add optional flags
[[ "$top_k" != "None" ]] && cmd+=" --top_k $top_k"
[[ -n "$baseline_flag" ]] && cmd+=" $baseline_flag" # Append baseline flag if set
[[ -n "$global_top_k" ]] && cmd+=" $global_top_k" # Append global_top_k flag if set

# Execute the constructed command
eval $cmd
