import torch

from .constants import PAD_IDX, BOS_IDX, EOS_IDX, SEP_IDX

from mamba_ssm.utils.generation import InferenceParams

def greedy_decode(model, device, max_len, src, src_padding_mask, start_symbol, dtype=torch.bfloat16, lm_head=False):
    """
    Performs greedy decoding to generate predictions.

    Args:
        src (Tensor): Source tensor of shape (batch_size, src_seq_len).
        src_padding_mask (Tensor): Source padding mask of shape (batch_size, src_seq_len).
        start_symbol (int): Start token index.

    Returns:
        Tensor: Generated token sequence of shape (batch_size, tgt_seq_len).
    """
    if lm_head:
        return greedy_decode_lmhead(model, device, max_len, src, src_padding_mask, start_symbol, dtype)

    src = src.to(device)
    src_padding_mask = src_padding_mask.to(device)
    batch_size = src.size(0)

    ys = torch.full((batch_size, 1), start_symbol, dtype=torch.long, device=device)
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

    with torch.no_grad():
        with torch.autocast(device_type='cuda', dtype=dtype):
            memory = model.encode(src, src_padding_mask)

            for _ in range(max_len):
                tgt_padding_mask = (ys != PAD_IDX)

                out = model.decode(ys, memory, tgt_padding_mask, src_padding_mask)

                prob = model.generator(out[:, -1, :])  # (batch_size, vocab_size)
                _, next_word = torch.max(prob, dim=1)       # (batch_size,)
                next_word = next_word.unsqueeze(1)          # (batch_size, 1)
                next_word[finished] = PAD_IDX


                ys = torch.cat([ys, next_word], dim=1)
                newly_finished = (next_word.squeeze(1) == EOS_IDX) | (next_word.squeeze(1) == SEP_IDX)
                finished = finished | newly_finished

                # Stop decoding if all sequences are finished
                # print(finished)
                if finished.all():
                    break
                # if ((next_word == EOS_IDX) | (next_word == SEP_IDX)).all():
                #     break

    return ys

# import torch
# # from mamba_ssm.utils.generation import InferenceParams # Assuming this import is available
# def greedy_decode_lmhead(model, device, max_len, src, src_padding_mask, start_symbol, dtype=torch.bfloat16):
# # def greedy_decode_lmhead(model, device, max_len, src, src_padding_mask, start_symbol, dtype=torch.bfloat16):
#     """
#     Efficient greedy decoding for a Mamba-based model using a key-value cache.

#     This implementation first processes the entire prompt to pre-fill the cache,
#     then generates subsequent tokens one by one for improved performance.

#     Args:
#         model: The Mamba model instance, which should have an `allocate_inference_cache` method.
#         device: The torch device to run generation on (e.g., "cuda" or "cpu").
#         max_len (int): The maximum number of new tokens to generate after the prompt.
#         src (Tensor): The input prompt tensor of shape (batch_size, src_len).
#         src_padding_mask: The padding mask for the source.
#         start_symbol: The start symbol (not used in this implementation).
#         dtype: The torch dtype for inference (e.g., torch.bfloat16).

#     Returns:
#         A torch.Tensor of shape (batch_size, src_len + generated_len) containing the
#         original prompt followed by the generated tokens.
#     """
#     # 1. Initialization
#     src = src.to(device)
#     batch_size, src_len = src.shape

#     # This tensor will store the full sequence (prompt + generation)
#     generated_ids = src.clone()

#     # 2. Allocate inference cache and parameters
#     # This is the core of the efficient implementation
#     cache = model.model.allocate_inference_cache(
#         batch_size=batch_size,
#         max_seqlen=src_len + max_len,
#         dtype=dtype,
#         device=device
#     )
#     inference_params = InferenceParams(
#         max_seqlen=src_len + max_len,
#         max_batch_size=batch_size,
#         key_value_memory_dict=cache,
#         seqlen_offset=0,  # We start processing from the beginning
#     )

#     # A boolean tensor to track which sequences in the batch are complete
#     finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

#     # 3. Generation Process
#     with torch.no_grad():
#         with torch.autocast(device_type='cuda', dtype=dtype):

#             # **Prefill Step**: Process the entire prompt to populate the cache
#             outputs = model.forward(
#                 input_ids=src,
#                 inference_params=inference_params
#             )
#             logits = outputs.logits

#             # Manually update the sequence offset after processing the prompt
#             inference_params.seqlen_offset += src_len

#             # Get the logits for the very next token following the prompt
#             next_token_logits = logits[:, -1, :]

#             # **Decoding Loop**: Generate subsequent tokens one by one
#             for _ in range(max_len):
#                 # Select the next token using greedy search (argmax)
#                 next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(1)

#                 # For sequences that have already finished, replace the new token with a pad token
#                 next_token[finished] = PAD_IDX

#                 # Append the newly generated token to our sequence
#                 generated_ids = torch.cat([generated_ids, next_token], dim=1)

#                 # Check for and update any sequences that just finished
#                 newly_finished = (next_token.squeeze(1) == EOS_IDX) | (next_token.squeeze(1) == SEP_IDX)
#                 finished |= newly_finished

#                 # If all sequences in the batch have finished, we can stop early
#                 if finished.all():
#                     break

#                 # **Cached Step**: Feed only the last token into the model for the next iteration
#                 # Create position_ids for the current step
#                 position_ids = torch.full(
#                     (batch_size, 1),
#                     inference_params.seqlen_offset,
#                     dtype=torch.long,
#                     device=device,
#                 )

#                 outputs = model.forward(
#                     input_ids=next_token,
#                     position_ids=position_ids,
#                     inference_params=inference_params,
#                     num_last_tokens=1
#                 )
#                 logits = outputs.logits

#                 # We processed one more token, so increment the offset by 1
#                 inference_params.seqlen_offset += 1

#                 # The output logits are for the *next* token to be generated
#                 next_token_logits = logits[:, -1, :]

#     return generated_ids[:, src_len:]

def greedy_decode_lmhead(model, device, max_len, src, src_padding_mask, start_symbol, dtype=torch.bfloat16):
    """
    Efficient greedy decoding for a Mamba-based seq2seq model using a key-value cache.

    For seq2seq tasks with unique mappings, this ensures proper conditioning on the source.

    Args:
        model: The Mamba model instance, which should have an `allocate_inference_cache` method.
        device: The torch device to run generation on (e.g., "cuda" or "cpu").
        max_len (int): The maximum number of new tokens to generate after the prompt.
        src (Tensor): The input prompt tensor of shape (batch_size, src_len).
        src_padding_mask: The padding mask for the source.
        start_symbol: The start symbol (not used in this implementation).
        dtype: The torch dtype for inference (e.g., torch.bfloat16).

    Returns:
        A torch.Tensor containing ONLY the generated tokens (without source tokens).
        Shape: (batch_size, generated_len)
    """
    # 1. Initialization
    src = src.to(device)
    sep_pos = (src == SEP_IDX).int().argmax(dim=1)
    max_sep_pos = sep_pos.max().item() + 1  # +1 to include the SEP_IDX

    # Truncate to the maximum separator position across the batch
    src = src[:, :max_sep_pos]


    batch_size, src_len = src.shape

    # For seq2seq, we need to ensure the model sees the full context properly
    # This tensor will store the full sequence (prompt + generation)
    generated_ids = src.clone()

    # 2. Allocate inference cache and parameters
    cache = model.model.allocate_inference_cache(
        batch_size=batch_size,
        max_seqlen=src_len + max_len,
        dtype=dtype,
        device=device
    )

    # CRITICAL: Start with clean cache state
    inference_params = InferenceParams(
        max_seqlen=src_len + max_len,
        max_batch_size=batch_size,
        key_value_memory_dict=cache,
        seqlen_offset=0,
    )

    # A boolean tensor to track which sequences in the batch are complete
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

    # Store only the generated tokens (separate from source)
    generated_tokens = []

    # 3. Generation Process
    with torch.no_grad():
        # Set model to eval mode to ensure consistent behavior
        model.eval()

        with torch.autocast(device_type='cuda', dtype=dtype):

            # **Prefill Step**: Process the entire source sequence to populate the cache
            # This is CRITICAL for seq2seq - the model must fully encode the source
            outputs = model.forward(
                input_ids=src,
                inference_params=inference_params
            )
            logits = outputs.logits

            # Update the sequence offset after processing the source
            inference_params.seqlen_offset = src_len

            # Get the logits for the first target token (conditioned on full source)
            next_token_logits = logits[:, -1, :]  # Last position logits

            # **Decoding Loop**: Generate target tokens one by one
            for step in range(max_len):
                # Use greedy decoding for deterministic unique mapping
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

                # For sequences that have finished, use pad token
                next_token[finished] = PAD_IDX

                # Store the generated token (target vocabulary)
                generated_tokens.append(next_token)

                # Append to full sequence for continued generation
                generated_ids = torch.cat([generated_ids, next_token], dim=1)

                # Check for sequences that just finished
                newly_finished = (next_token.squeeze(1) == EOS_IDX) | (next_token.squeeze(1) == SEP_IDX)
                finished |= newly_finished

                # Early stopping if all sequences finished
                if finished.all():
                    break

                # **Cached Step**: Continue generation with the new token
                # Position IDs should reflect the actual position in the full sequence
                position_ids = torch.full(
                    (batch_size, 1),
                    src_len + step,  # Current position in the full sequence
                    dtype=torch.long,
                    device=device,
                )

                # Forward pass with only the last token
                outputs = model.forward(
                    input_ids=next_token,
                    position_ids=position_ids,
                    inference_params=inference_params,
                    num_last_tokens=1
                )
                logits = outputs.logits

                # Update offset
                inference_params.seqlen_offset += 1

                # Get logits for the next token
                next_token_logits = logits[:, -1, :]

    # Return only the generated tokens (target vocabulary)
    if generated_tokens:
        return torch.cat(generated_tokens, dim=1)
    else:
        # If no tokens were generated, return empty tensor
        return torch.empty(batch_size, 0, dtype=torch.long, device=device)