#!/bin/bash

#SBATCH --output=/scratch/yy2694/llm-memory/slurm_outputs/%x_%j.out
#SBATCH --error=/scratch/yy2694/llm-memory/slurm_outputs/%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yy2694@nyu.edu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=8:00:00
#SBATCH --mem=128GB
#SBATCH --gres=gpu:a100:2
#SBATCH --job-name=llm-memory-scratch-2.7b

module purge

singularity exec --nv \
    --overlay /scratch/yy2694/overlay-50G-10M.ext3:ro \
    /scratch/work/public/singularity/cuda11.3.0-cudnn8-devel-ubuntu20.04.sif \
    /bin/bash -c "source /ext3/env.sh; cd /scratch/yy2694/llm-memory; accelerate launch --config_file default_config.yaml --num_cpu_threads_per_process 4 train_generate.py --model_name_or_path "facebook/opt-2.7b" --train_file "data/llm-experiment-data/expt1/seen_data_0.json" --seen_file "data/llm-experiment-data/expt1/seen_data_0.json" --per_device_train_batch_size 4 --per_device_eval_batch_size 1 --learning_rate 0.0001 --output_dir checkpoints/opt-2.7b-scratch --save_prefix batch4_gpu2 --block_size 128 --num_train_epochs 100 --overwrite_cache --checkpointing_steps epoch --no-use_pretrained_weights --eval_freq 5"
