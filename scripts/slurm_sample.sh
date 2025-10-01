#!/bin/bash

#SBATCH -A m4392
#SBATCH -C gpu&hbm80g
#SBATCH -G 4
#SBATCH -q regular
#SBATCH -t 15:00:00
#SBATCH -n 1
#SBATCH -c 128
#SBATCH --output="/pscratch/sd/p/pr4santh/slurm_logs_out/slurm-%j.out"
#SBATCH --error="/pscratch/sd/p/pr4santh/slurm_logs_err/slurm-%j.out"
#SBATCH --mail-user=prasanthnaidukaraka@gmail.com
#SBATCH --mail-type=ALL

conda init
conda activate ssm_env

torchrun --nnodes=1 --nproc_per_node=4 --master_port=29501 -m SYMBA_SSM.main --project_name EW \
        --run_name EW_2_3_e512_d2048 \
        --model_name EW_2_3_e512_d2048 \
        --root_dir /pscratch/sd/p/pr4santh/results/symba \
        --data_dir /pscratch/sd/p/pr4santh/Data/permutation_free/EW-2-to-3/ --device cuda \
        --epochs 70 \
        --training_batch_size 32 \
        --valid_batch_size 32 \
        --num_workers 128 \
        --embedding_size 512 \
        --ff_dims 4096 \
        --nhead 8 \
        --num_encoder_layers 3 \
        --num_decoder_layers 4 \
        --warmup_ratio 0.1 \
        --dropout 0.2 \
        --weight_decay 3e-4 \
        --optimizer_lr 0.0005 \
        --src_max_len 512 \
        --tgt_max_len 2048 \
        --curr_epoch 0 \
        --save_last \
        --filter_len \
        --save_freq 5 \
        --to_replace \
        --index_pool_size 1024


