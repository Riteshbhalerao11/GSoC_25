import os

import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from .fn_utils import (
    decode_sequence,
    create_mask,
    generate_unique_random_integers,
    get_model
)

from .inference import greedy_decode, beam_search_decode
from .constants import PAD_IDX



def collate_fn(batch):
    """
    batch: list of tuples (src_tensor, original_tokens)
    Returns:
        src: padded src batch tensor
        original_tokens: list of original token tensors
    """
    src_tensors = [example[0] for example in batch]
    src = pad_sequence(src_tensors, padding_value=PAD_IDX, batch_first=True)
    original_tokens = [example[1] for example in batch]

    return src, original_tokens



class Predictor:
    def __init__(self, config, load_best=True, epoch=None):
        self.model = get_model(config)
        self.checkpoint = (
            f"{config.model_name}_best.pth"
            if load_best else f"{config.model_name}_ep{epoch + 1}.pth"
        )
        if hasattr(config, 'finetune') and config.finetune:
            self.finetune_dir = os.path.join(config.root_dir, "finetune")
            self.path = os.path.join(self.finetune_dir, self.checkpoint) 
            print("Loading finetuned model from:", self.path)
        else:
            self.path = os.path.join(config.root_dir, self.checkpoint)
        self.device = config.device
        self.dtype = {
            'bfloat16': torch.bfloat16,
            'float16': torch.float16,
            'float32': torch.float32
        }[config.dtype]

        state = torch.load(self.path, map_location=self.device)
        self.model.load_state_dict(state['state_dict'])
        self.model.to(self.device)
        self.max_len = config.tgt_max_len
        self.is_beamsearch = config.is_beamsearch

        if self.is_beamsearch:
            assert config.beam_width > 0, "Beam width must be greater than 0 for beam search decoding."
            self.beam_width = config.beam_width

        print(f"Using epoch {state['epoch']} model for predictions.")

    def predict_batch(self, batch, vocab, raw_tokens=False):
        """
        Args:
            batch: (src_tensor, original_tokens)
        """
        self.model.eval()
        src, original_tokens = batch
        batch_size = src.size(0)

        # Start symbols: first token from each original sequence
        start_symbols = torch.stack([tokens[0] for tokens in original_tokens], dim=0).reshape(batch_size, 1)

        src_padding_mask, _ = create_mask(
            src, torch.zeros((batch_size, 1), dtype=src.dtype, device=self.device)
        )

        with torch.no_grad():
            if self.is_beamsearch:
                tgt_tokens = beam_search_decode(
                    self.model, self.device, self.max_len,
                    src, src_padding_mask,
                    start_symbols=start_symbols,
                    dtype=self.dtype,
                    beam_width=self.beam_width
                )
            else:
                tgt_tokens = greedy_decode(
                    self.model, self.device, self.max_len,
                    src, src_padding_mask,
                    start_symbols=start_symbols,
                    dtype=self.dtype
                )

        if raw_tokens:
            return [(gt, pred) for gt, pred in zip(original_tokens, tgt_tokens)]
        return [vocab.decode(seq.tolist()) for seq in tgt_tokens]


def sequence_accuracy(config, test_ds, vocab, load_best=True, epoch=None, return_incorrect=False):
    test_size = config.test_size
    if hasattr(config, 'finetune') and config.finetune:
        test_size = 5000
    predictor = Predictor(config, load_best, epoch)
    num_samples = 10 if config.debug else test_size

    if num_samples >= len(test_ds):
        eval_indices = list(range(len(test_ds)))
    else:
        eval_indices = generate_unique_random_integers(num_samples, start=0, end=len(test_ds))

    eval_subset = torch.utils.data.Subset(test_ds, eval_indices)

    dataloader = DataLoader(
        eval_subset,
        batch_size=config.test_batch_size,
        shuffle=False,          
        collate_fn=collate_fn
    )

    count, total = 0, 0
    incorrect_seqs, incorrect_idxs = [], []

    pbar = tqdm(dataloader, desc="Seq_Acc_Cal")
    for batch_start, batch in enumerate(pbar):
        raw_pairs = predictor.predict_batch(batch, vocab, raw_tokens=True)

        for j, (gt_tokens, pred_tokens) in enumerate(raw_pairs):
            gt = decode_sequence(gt_tokens.detach().cpu().tolist(), vocab)
            pred = decode_sequence(pred_tokens.detach().cpu().tolist(), vocab)

            if gt == pred:
                count += 1
            elif return_incorrect:
                incorrect_seqs.append((gt, pred))
    
                subset_idx = batch_start * config.test_batch_size + j
                incorrect_idxs.append(eval_indices[subset_idx])

            total += 1

        pbar.set_postfix(seq_accuracy=count / total)

    if return_incorrect:
        return count / total, incorrect_seqs, incorrect_idxs

    return count / total