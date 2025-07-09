import os


import torch
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from .fn_utils import (
    decode_sequence,
    create_mask,
    generate_unique_random_integers,
    get_model
)

from .inference import greedy_decode
from .constants import PAD_IDX


class Predictor:
    def __init__(self, config, load_best=True, epoch=None):
        self.model = get_model(config)
        self.checkpoint = (
            f"{config.model_name}_best.pth"
            if load_best else f"{config.model_name}_ep{epoch + 1}.pth"
        )
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

        print(f"Using epoch {state['epoch']} model for predictions.")

    def predict_batch(self, batch, vocab, raw_tokens=False):
        """
        Args:
            batch: list of test examples (src_tensor, original_tokens)
            vocab: tokenizer
            raw_tokens: if True, return (ground_truth, prediction) token lists

        Returns:
            List of decoded strings or token tuples
        """
        self.model.eval()
        batch_size = len(batch)
        src = pad_sequence([example[0] for example in batch], padding_value=PAD_IDX, batch_first=True)
        original_tokens = [example[1] for example in batch]
        start_symbols = torch.stack([example[1][0] for example in batch],dim=0).reshape(batch_size, 1)

        src_padding_mask, _ = create_mask(
            src, torch.zeros((batch_size, 1), dtype=self.dtype, device=self.device)
        )
        with torch.no_grad():
            tgt_tokens = greedy_decode(
                self.model, self.device, self.max_len,
                src, src_padding_mask,
                start_symbols=start_symbols,
                dtype=self.dtype
            )

        if raw_tokens:
            return [(gt, pred) for gt, pred in zip(original_tokens, tgt_tokens)]

        return [vocab.decode(seq) for seq in tgt_tokens]



def sequence_accuracy(config, test_ds, vocab, load_best=True, epoch=None, test_size=1000):
    predictor = Predictor(config, load_best, epoch)
    test_batch_size = config.test_batch_size
    num_samples = 10 if config.debug else test_size

    random_idx = generate_unique_random_integers(num_samples, start=0, end=len(test_ds))
    count = 0
    total = 0

    pbar = tqdm(range(0, num_samples, test_batch_size))
    pbar.set_description("Seq_Acc_Cal")

    for i in pbar:
        batch_indices = random_idx[i:min(i + test_batch_size, num_samples)]
        batch = [test_ds[idx] for idx in batch_indices]

        raw_pairs = predictor.predict_batch(batch, vocab, raw_tokens=True)

        for gt_tokens, pred_tokens in raw_pairs:
            gt = decode_sequence(gt_tokens.detach().cpu().tolist(), vocab)
            pred = decode_sequence(pred_tokens.detach().cpu().tolist(), vocab)

            if gt == pred:
                count += 1
            total += 1

        pbar.set_postfix(seq_accuracy=count / total)

    return count / total