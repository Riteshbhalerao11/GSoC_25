import torch
from .constants import PAD_IDX, EOS_IDX, SEP_IDX
import torch.nn.functional as F
import contextlib

def sample_from_logits(logits, temperature=1.0, top_k=0):
    """
    Sample from logits with temperature and top-k filtering.
    Args:
        logits: (B, vocab)
        temperature: float
        top_k: int, if > 0 applies top-k filtering
    Returns:
        next_token: (B, 1)
    """
    logits = logits / temperature
    if top_k > 0:
        topk_vals, topk_idx = torch.topk(logits, top_k)
        probs = F.softmax(topk_vals, dim=-1)
        sampled = torch.multinomial(probs, num_samples=1)
        next_token = topk_idx.gather(1, sampled)
    else:
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
    return next_token



def greedy_decode(model, device, max_len, src, src_padding_mask, start_symbols, 
                  dtype=torch.bfloat16, temperature=0, top_k=0):
    """
    Efficient greedy decoding with per-sample EOS and dynamic shrinking.
    Args:
        src: (B, S)
        src_padding_mask: (B, S)
        start_symbols: (B, 1)
        eval_mode: If True, disables gradient tracking.

    Returns:
        - padded: Tensor of shape (B, <= max_len + 1), right-padded with PAD_IDX
        - logprobs (optional): Tensor of shape (B,), sum of logprobs if eval_mode=False
    """
    src = src.to(device)
    start_symbols = start_symbols.to(device)
    src_padding_mask = src_padding_mask.to(device)
    batch_size = src.size(0)


    with torch.no_grad():
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

            if temperature <= 0:
                 _, next_token = torch.max(logits, dim=-1)  # greedy
            else:
                next_token = sample_from_logits(logits, temperature, top_k).squeeze(1)

            next_word = next_token.unsqueeze(1)
            current_ys_alive = torch.cat([current_ys_alive, next_word], dim=1)

            # EOS check
            eos_hit = (next_token == EOS_IDX) | (next_token == SEP_IDX)
            finished_idx = alive_idx[eos_hit]

            for i, idx in enumerate(alive_idx.tolist()):
                if eos_hit[i]:
                    final_outputs[idx] = current_ys_alive[i].clone()

            finished_flag[finished_idx] = True
            alive_idx = (~finished_flag).nonzero(as_tuple=False).squeeze(-1)

            if alive_idx.numel() == 0:
                break

            current_ys_alive = current_ys_alive[~eos_hit]
            current_ys = torch.full((batch_size, current_ys_alive.shape[1]), PAD_IDX,
                                    dtype=torch.long, device=device)
            current_ys[alive_idx] = current_ys_alive

        # Handle unfinished
        for idx in alive_idx.tolist():
            final_outputs[idx] = current_ys[idx]

    # Pad
    final_len = max(seq.size(0) for seq in final_outputs)
    padded = torch.full((batch_size, final_len), PAD_IDX, dtype=torch.long, device=device)
    for i, seq in enumerate(final_outputs):
        padded[i, :seq.size(0)] = seq

    return padded
    

# BEAM SEARCH TO BE TESTED: BUGGY

def beam_search_decode(
    model, device, max_len, src, src_padding_mask, start_symbols,
    beam_width=5, dtype=torch.bfloat16
):
\
    batch_size = src.size(0)
    src, src_padding_mask, start_symbols = map(lambda x: x.to(device), [src, src_padding_mask, start_symbols])

    memory = model.encode(src, src_padding_mask)  # (batch_size, S, H)
    seq_len = memory.size(1)
    embed_dim = memory.size(2)

    memory = memory.unsqueeze(1).expand(batch_size, beam_width, -1, -1).reshape(batch_size * beam_width, seq_len, embed_dim)
    src_padding_mask = src_padding_mask.unsqueeze(1).expand(batch_size, beam_width, -1).reshape(batch_size * beam_width, -1)



    tgt_padding_mask = (start_symbols != PAD_IDX)
    logits = model.generator(model.decode(start_symbols, memory[::beam_width], tgt_padding_mask, src_padding_mask[::beam_width]))
    log_probs = torch.log_softmax(logits, dim=-1)  # (batch_size, vocab_size)

    topk_log_probs, topk_indices = log_probs.topk(beam_width, dim=-1)  # (batch_size, beam_width)


    beams = start_symbols.repeat_interleave(beam_width, dim=0)  # (batch_size * beam_width, 1)
    beams = torch.cat([beams, topk_indices.view(-1, 1)], dim=1)  # (batch_size * beam_width, 2)

    beam_scores = topk_log_probs.squeeze(1)  # (batch_size, beam_width)
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)


    for step in range(2, max_len + 1):
        tgt_padding_mask = (beams != PAD_IDX)
        logits = model.generator(model.decode(beams, memory, tgt_padding_mask, src_padding_mask))
        log_probs = torch.log_softmax(logits[:, -1, :], dim=-1)  # (B*beam_width, vocab)

        log_probs = log_probs.view(batch_size, beam_width, -1)

        total_scores = beam_scores.unsqueeze(-1) + log_probs  # (batch_size, beam_width, vocab)
        flat_scores = total_scores.view(batch_size, -1)

        topk_scores, topk_indices = flat_scores.topk(beam_width, dim=-1)
        next_beam_ids = topk_indices // log_probs.shape[-1]
        next_token_ids = topk_indices % log_probs.shape[-1]

        new_beams = []
        for i in range(batch_size):
            for j in range(beam_width):
                beam_idx = i * beam_width + next_beam_ids[i, j]
                token = next_token_ids[i, j]
                new_seq = torch.cat([beams[beam_idx], token.view(1)])
                new_beams.append(new_seq)
        beams = torch.stack(new_beams, dim=0)  # (batch_size * beam_width, step+1)
        beam_scores = topk_scores  # (batch_size, beam_width)

 
        eos_mask = (beams[:, -1] == EOS_IDX) | (beams[:, -1] == SEP_IDX)  # (batch_size * beam_width)
        eos_mask = eos_mask.view(batch_size, beam_width)
        new_finished = eos_mask.any(dim=1)  # (batch_size,)
        finished |= new_finished

        if finished.all():
            break

    best_beam_ids = beam_scores.argmax(dim=1)  # (batch_size,)
    output = []
    for i in range(batch_size):
        best_beam_idx = i * beam_width + best_beam_ids[i]
        if best_beam_idx >= beams.size(0):
            print(f"Warning: beam idx {best_beam_idx} out of bounds, using fallback.")
            best_beam_idx = i * beam_width  # fallback to first beam
        output.append(beams[best_beam_idx])

    max_len = max(seq.size(0) for seq in output)
    padded = torch.full((batch_size, max_len), PAD_IDX, dtype=torch.long, device=device)
    for i, seq in enumerate(output):
        padded[i, :seq.size(0)] = seq

    return padded
