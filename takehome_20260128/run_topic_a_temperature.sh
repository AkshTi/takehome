#!/bin/bash
#SBATCH -J topic_a_temperature
#SBATCH -p mit_normal_gpu
#SBATCH -t 03:00:00
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -G h200:1
#SBATCH --chdir=/home/akshatat/takehome/takehome_20260128
#SBATCH -o /home/akshatat/takehome/logs/%x_%j.out
#SBATCH -e /home/akshatat/takehome/logs/%x_%j.err

mkdir -p /home/akshatat/takehome/logs

source activate distill

echo "Job ID:     $SLURM_JOB_ID"
echo "Node:       $SLURMD_NODENAME"
echo "GPU:        $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"

python topic_a_temperature.py

echo "End time: $(date)"
