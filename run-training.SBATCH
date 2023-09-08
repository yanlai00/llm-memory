#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00
#SBATCH --mem=128GB
#SBATCH --gres=gpu:2
#SBATCH --job-name=llm-memory

module purge

singularity exec --nv \
    --overlay /scratch/yy2694/overlay-50G-10M.ext3:ro \
    /scratch/work/public/singularity/cuda11.3.0-cudnn8-devel-ubuntu20.04.sif \
    /bin/bash -c "source /ext3/env.sh; cd /scratch/yy2694/llm-memory; accelerate launch --config_file accelerate_config.yaml --num_cpu_threads_per_process 4 train.py --model_name_or_path "facebook/opt-1.3b" --train_file "data/llm-experiment-data/expt1/seen_data_0.json" --per_device_train_batch_size 4 --learning_rate 0.00001 --output_dir checkpoints/opt-1.3b --save_prefix train_epoch10_batch16 --block_size 128 --num_train_epochs 10 --overwrite_cache --checkpointing_steps epoch"
