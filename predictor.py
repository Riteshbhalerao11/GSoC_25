import os

import torch
from tqdm import tqdm

from .fn_utils import (
    decode_sequence,
    create_mask,
    generate_unique_random_integers,
    get_model
)
from .constants import PAD_IDX, BOS_IDX, EOS_IDX


class Predictor:
    """
    Class for generating predictions using a trained model and greedy decoding.

    Args:
        config (object): Configuration object containing model and inference settings.
        load_best (bool, optional): Whether to load the best model. Defaults to True.
        epoch (int, optional): Epoch number to load a specific checkpoint.

    Attributes:
        model (Model): Trained model for prediction.
        path (str): Path to the trained model checkpoint.
        device (str): Device for inference.
        checkpoint (str): Model checkpoint filename.
        max_len (int): Maximum target sequence length for inference.
    """

    def __init__(self, config, load_best=True, epoch=None):
        self.model = get_model(config)
        self.checkpoint = (
            f"{config.model_name}_best.pth"
            if load_best else f"{config.model_name}_ep{epoch + 1}.pth"
        )
        self.path = os.path.join(config.root_dir, self.checkpoint)
        self.device = config.device
        if config.dtype == 'bfloat16':
            self.dtype = torch.bfloat16
        elif config.dtype == 'float32':
            self.dtype = torch.float32
        elif config.dtype == 'float16':
            self.dtype = torch.float16

        # Load model checkpoint
        state = torch.load(self.path, map_location=self.device)
        self.model.load_state_dict(state['state_dict'])
        self.model.to(self.device)
        self.max_len = config.tgt_max_len

        print(f"Using epoch {state['epoch']} model for predictions.")

    def greedy_decode(self, src, src_padding_mask, start_symbol):
        """
        Performs greedy decoding to generate predictions.

        Args:
            src (Tensor): Source tensor of shape (batch_size, src_seq_len).
            src_padding_mask (Tensor): Source padding mask of shape (batch_size, src_seq_len).
            start_symbol (int): Start token index.

        Returns:
            Tensor: Generated token sequence of shape (batch_size, tgt_seq_len).
        """
        src = src.to(self.device)
        src_padding_mask = src_padding_mask.to(self.device)
        batch_size = src.size(0)

        ys = torch.full((batch_size, 1), start_symbol, dtype=torch.long, device=self.device)

        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=self.dtype):
                memory = self.model.encode(src, src_padding_mask)

                for _ in range(self.max_len):
                    tgt_padding_mask = (ys != PAD_IDX)

                    out = self.model.decode(ys, memory, tgt_padding_mask, src_padding_mask)

                    prob = self.model.generator(out[:, -1, :])  # (batch_size, vocab_size)
                    _, next_word = torch.max(prob, dim=1)       # (batch_size,)
                    next_word = next_word.unsqueeze(1)          # (batch_size, 1)

                    ys = torch.cat([ys, next_word], dim=1)      

                    if (next_word == EOS_IDX).all():
                        break

        return ys

    def predict(self, test_example, vocab, raw_tokens=False):
        """
        Generates predictions for a given test example.

        Args:
            test_example (tuple): Tuple containing source tensor and original tokens.
            itos (dict): Index-to-string vocabulary mapping.
            raw_tokens (bool, optional): Whether to return raw token outputs. Defaults to False.

        Returns:
            str or tuple: Decoded equation or tuple of original and generated tokens.
        """
        self.model.eval()

        src = test_example[0].unsqueeze(0)
        src_padding_mask, _ = create_mask(
            src, torch.zeros((1, 1), dtype=self.dtype, device=self.device)
        )

        tgt_tokens = self.greedy_decode(
            src, src_padding_mask, start_symbol=BOS_IDX
        ).flatten()

        if raw_tokens:
            return test_example[1], tgt_tokens

        return ''.join(vocab.decode(tgt_tokens))


def sequence_accuracy(config, test_ds, vocab, load_best=True, epoch=None, test_size=100):
    """
    Calculate the sequence accuracy.

    Args:
        config (object): Configuration for inference.
        test_ds (list): Dataset for testing.
        tgt_itos (dict): Index-to-token mapping.
        load_best (bool, optional): Whether to load the best model. Defaults to True.
        epoch (int, optional): Specific epoch to load. Defaults to None.
        test_size (int, optional): Number of test samples to evaluate. Defaults to 100.

    Returns:
        float: Sequence accuracy.
    """
    predictor = Predictor(config, load_best, epoch)
    count = 0
    num_samples = 10 if config.debug else test_size

    random_idx = generate_unique_random_integers(
        num_samples, start=0, end=len(test_ds)
    )
    length = len(random_idx)

    pbar = tqdm(range(length))
    pbar.set_description("Seq_Acc_Cal")

    for i in pbar:
        original_tokens, predicted_tokens = predictor.predict(
            test_ds[random_idx[i]], vocab, raw_tokens=True
        )
        original_tokens = original_tokens.detach().numpy().tolist()
        predicted_tokens = predicted_tokens.detach().cpu().numpy().tolist()

        original = decode_sequence(original_tokens, vocab)
        predicted = decode_sequence(predicted_tokens, vocab)

        if original == predicted:
            count += 1

        pbar.set_postfix(seq_accuracy=count / (i + 1))

    return count / length
