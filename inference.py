import torch
from .constants import PAD_IDX, EOS_IDX, SEP_IDX

def greedy_decode(model, device, max_len, src, src_padding_mask, start_symbols, dtype=torch.bfloat16):
    """
    Efficient greedy decoding with per-sample EOS and dynamic shrinking.
    Preserves output order.

    Args:
        src: (B, S)
        src_padding_mask: (B, S)
        start_symbols: (B, 1)

    Returns:
        Tensor of shape (B, <= max_len + 1), right-padded with PAD_IDX
    """
    src = src.to(device)
    start_symbols = start_symbols.to(device)
    src_padding_mask = src_padding_mask.to(device)
    batch_size = src.size(0)

    with torch.no_grad():
        with torch.autocast(device_type='cuda', dtype=dtype):
            memory = model.encode(src, src_padding_mask)

            alive_idx = torch.arange(batch_size, device=device)
            current_ys = start_symbols.clone()  # (B, 1)
            final_outputs = [None] * batch_size
            finished_flag = torch.zeros(batch_size, dtype=torch.bool, device=device)

            for _ in range(max_len):

                current_ys_alive = current_ys[alive_idx]
                tgt_padding_mask = (current_ys_alive != PAD_IDX)
                current_memory = memory[alive_idx]
                current_src_mask = src_padding_mask[alive_idx]

                out = model.decode(current_ys_alive, current_memory, tgt_padding_mask, current_src_mask)
                logits = model.generator(out[:, -1, :])  # (alive, vocab)
                next_word = torch.argmax(logits, dim=-1).unsqueeze(1)  # (alive, 1)

                # Append new tokens
                current_ys_alive = torch.cat([current_ys_alive, next_word], dim=1)

                # check terminal condition
                eos_hit = (next_word.squeeze(1) == EOS_IDX) | (next_word.squeeze(1) == SEP_IDX)
                finished_idx = alive_idx[eos_hit]

                for i, idx in enumerate(alive_idx.tolist()):
                    if eos_hit[i]:
                        final_outputs[idx] = current_ys_alive[i].clone()

                # Update alive idxs
                finished_flag[finished_idx] = True
                alive_idx = (~finished_flag).nonzero(as_tuple=False).squeeze(-1)

                if alive_idx.numel() == 0:
                    break

                current_ys_alive = current_ys_alive[~eos_hit]

                # Prepare next batch
                current_ys = torch.full((batch_size, current_ys_alive.shape[1]), PAD_IDX, dtype=torch.long, device=device)
                current_ys[alive_idx] = current_ys_alive

            # Fill in unfinished sequences
            for idx in alive_idx.tolist():
                final_outputs[idx] = current_ys[idx]

    # Pad to max length
    final_len = max(seq.size(0) for seq in final_outputs)
    padded = torch.full((batch_size, final_len), PAD_IDX, dtype=torch.long, device=device)
    for i, seq in enumerate(final_outputs):
        padded[i, :seq.size(0)] = seq

    return padded
