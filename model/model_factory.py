from .mamba_encdec import MambaEncDec

# module_mapping = {
#             'FFN':FeedForwardWrapper,
#             'FMHSA':FlashSelfAttentionWrapper,
#             'FXA':FlashCrossAttentionWrapper,
#             'FSWA':SlidingAttentionWrapper,
#             'Mamba':MixerModel
#         }


def construct_model(config):
    # if config.not_ssm:
    #     encoder_layers = ['FMHSA', 'FFN'] * (config.num_encoder_layers)
    #     decoder_layers = ['FXA', 'FFN'] * (config.num_decoder_layers)

    # else:

    if config.lm_head:
        from .mamba_dec import MambaMT
        mamba_config = {
            "d_model": config.embedding_size,
            "n_layer": (config.num_encoder_layers + config.num_decoder_layers + 1),
            "rms_norm": True,
            "fused_add_norm": True,
            "pad_vocab_size_multiple": 1
            # "use_fast_path": False,
        }

        model = MambaMT(
            **mamba_config,
            config = mamba_config,
            src_vocab_size=config.src_voc_size,
            tgt_vocab_size=config.tgt_voc_size,
            precision="bf16-mixed"
            # tgt_vocab_size=39,
        )
        return model

    if config.vanilla_transformer:
        encoder_layers = ['FMHSA', 'FFN'] * (config.num_encoder_layers)
        decoder_layers = ['FXA', 'FFN'] * (config.num_decoder_layers)

    else:

        encoder_layers = ['Mamba'] * 2 * (config.num_encoder_layers - 1) + ['FMHSA', 'FFN']
        decoder_layers = ['FXA', 'FFN'] + ['Mamba'] * 2 * (config.num_decoder_layers - 2) + ['FXA', 'FFN']


    # encoder_layers = ['FMHSA', 'FFN'] * (config.num_encoder_layers)
    # decoder_layers = ['FXA', 'FFN'] * (config.num_decoder_layers)

    mamba_config = {
        "enc_n_layer": config.num_encoder_layers * 2,
        "d_model": config.embedding_size,
        "dec_n_layer": config.num_decoder_layers * 2,
        "rms_norm": True,
        "fused_add_norm": True,
        "use_fast_path": False,
        "encoder_layer_list": encoder_layers,
        "decoder_layer_list": decoder_layers,
        # "learning_rate": config.learning_rate,
        # "warmup_steps": config.warmup_steps,
        # "weight_decay": config.weight_decay,
        # "devices": config.devices
    }
    # mamba_config = {
    #     "enc_n_layer": config.num_encoder_layers,
    #     "d_model": config.d_model,
    #     "n_layer": config.n_layer,
    #     "rms_norm": config.rms_norm,
    #     "fused_add_norm": config.fused_add_norm,
    #     "use_fast_path": config.use_fast_path,
    #     "learning_rate": config.learning_rate,
    #     "warmup_steps": config.warmup_steps,
    #     "weight_decay": config.weight_decay,
    #     "devices": config.devices
    # }

    model = MambaEncDec(
        **mamba_config,
        config = mamba_config,
        src_vocab_size=config.src_voc_size,
        tgt_vocab_size=config.tgt_voc_size
    )

    return model