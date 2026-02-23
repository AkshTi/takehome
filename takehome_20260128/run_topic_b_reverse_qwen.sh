#!/bin/bash
#SBATCH -J topic_b_reverse_qwen
#SBATCH -p mit_normal_gpu
#SBATCH -t 01:00:00
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -G h200:1
#SBATCH --chdir=/home/akshatat/distillation/takehome/takehome_20260128
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

mkdir -p /home/akshatat/distillation/takehome/takehome_20260128/logs

echo "Job ID:     $SLURM_JOB_ID"
echo "Node:       $SLURMD_NODENAME"
echo "GPU:        $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"

python topic_b_reverse_qwen.py

echo "End time: $(date)"
