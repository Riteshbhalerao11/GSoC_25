torchrun --nnodes=1 --nproc_per_node=1 --master_port=29501 -m SYMBA_SSM.main --project_name EW \
        --run_name EW_2_2_baseline \
        --model_name enc_dec \
        --root_dir /pscratch/sd/p/pr4santh/results/symba \
        --data_dir /pscratch/sd/p/pr4santh/Data/permutation_free/EW-2-to-2/ \
        --device cuda \
        --epochs 50 \
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
        --src_max_len 4096 \
        --tgt_max_len 4096 \
        --curr_epoch 0 \
        --save_last \
        --filter_len \
        --save_freq 5 \
        --to_replace \
        --index_pool_size 1024
        # --truncate
        # --debug \
