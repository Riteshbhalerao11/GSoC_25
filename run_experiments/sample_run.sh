#!/bin/bash

#SBATCH -A <account>
#SBATCH -C gpu
#SBATCH -G 2
#SBATCH -q shared
#SBATCH -t 24:00:00
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --output="$SCRATCH/flash_logs/slurm-%j.out"
#SBATCH --error="$SCRATCH/flash_logs/slurm-%j.out"
#SBATCH --mail-user=<you@example.com>
#SBATCH --mail-type=FAIL,END

# Optional: load site modules / activate conda env
module load pytorch

# or
# source $HOME/miniconda3/bin/conda activate flash

torchrun --standalone --nproc_per_node 2 -m Flash.main \
  --project_name "flash_sample" \
  --run_name "demo_${SLURM_JOB_ID:-local}" \
  --model_name "flash_transformer_base" \
  --root_dir "$SCRATCH/Flash/checkpoints/sample_run" \
  --data_dir "$SCRATCH/Flash/data/sample_dataset/" \
  --device cuda \
  --epochs 10 \
  --training_batch_size 32 \
  --valid_batch_size 32 \
  --num_workers 8 \
  --embedding_size 512 \
  --ff_dims 4096 \
  --nhead 8 \
  --num_encoder_layers 3 \
  --num_decoder_layers 3 \
  --warmup_ratio 0.05 \
  --weight_decay 0.01 \
  --clip_grad_norm 1 \
  --dropout 0.1 \
  --src_max_len 512 \
  --tgt_max_len 1024 \
  --curr_epoch 0 \
  --optimizer_lr 5e-5 \
  --train_shuffle \
  --pin_memory \
  --world_size 2 \
  --save_freq 5 \
  --test_freq 5 \
  --seed 42 \
  --log_freq 20 \
  --save_last \
  --save_limit 3