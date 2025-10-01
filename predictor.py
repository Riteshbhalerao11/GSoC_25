import os
import csv
import torch
from tqdm import tqdm

from .fn_utils import (
    decode_sequence,
    collate_fn,
    create_mask,
    generate_unique_random_integers,
    get_model
)

from .inference import greedy_decode

from torch.utils.data import DataLoader, Subset

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
            f"{config.model_name}/best.pth"
            if load_best else f"{config.model_name}/ep{epoch + 1}.pth"
        )
        self.lm_head = config.lm_head
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

        tgt_tokens = greedy_decode(self.model, self.device, self.max_len,
            src, src_padding_mask, test_example[1][0], self.dtype).flatten()

        if raw_tokens:
            return test_example[1], tgt_tokens

        return ''.join(vocab.decode(tgt_tokens))

    def predict_batch(self, batch, vocab, raw_tokens=False):
        """
        Generates predictions for a batch of examples using batched greedy decoding.

        Args:
            batch (tuple): A batch from DataLoader — (src_batch, original_tokens_batch)
            vocab (Vocab): Vocabulary object with `decode` method.
            raw_tokens (bool): If True, returns token tensors instead of decoded strings.

        Returns:
            tuple: (original_tokens_batch, predicted_tokens_batch)
                - if raw_tokens=True: both are lists of token tensors
                - else: both are lists of decoded strings
        """
        self.model.eval()

        src_batch = batch[0].to(self.device)
        original_tokens_batch = batch[1]  # List of tuples of tokens (one per example)

        # Create padding mask for batch
        src_padding_mask, _ = create_mask(
            src_batch,
            torch.zeros((src_batch.shape[0], 1), dtype=self.dtype, device=self.device)
        )

        # Collect BOS tokens for each sample in batch
        bos_tokens = [item[0] for item in original_tokens_batch]

        # Assume greedy_decode can accept BOS tokens list
        # Shape: (batch_size, seq_len)
        tgt_tokens_batch = greedy_decode(
            self.model,
            self.device,
            self.max_len,
            src_batch,
            src_padding_mask,
            bos_tokens[0],
            self.dtype,
            self.lm_head
        )

        if raw_tokens:
            return original_tokens_batch, tgt_tokens_batch

        decoded_predictions = [''.join(vocab.decode(pred)) for pred in tgt_tokens_batch]
        return decoded_predictions


def sequence_accuracy(config, test_ds, vocab, load_best=True, epoch=None, on_all=False, test_size=300, batch_size=16, save_predictions=True):
    """
    Calculate sequence accuracy and optionally save predictions to CSV.

    Args:
        config: Configuration object
        test_ds: Test dataset
        vocab: Vocabulary object
        load_best: Whether to load best model
        epoch: Specific epoch to load
        on_all: Whether to evaluate on all data
        test_size: Number of samples to test
        batch_size: Batch size for inference
        save_predictions: Whether to save predictions to CSV

    Returns:
        float: Sequence accuracy
    """
    predictor = Predictor(config, load_best, epoch)
    count = 0
    num_samples = 10 if config.debug else test_size
    num_samples = min(num_samples, len(test_ds))
    random_idx = generate_unique_random_integers(
        num_samples, start=0, end=len(test_ds)
    )

    if on_all:
        sub_dataset = test_ds
    else:
        sub_dataset = Subset(test_ds, random_idx)

    dataloader = DataLoader(
        sub_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )

    pbar = tqdm(dataloader)
    pbar.set_description("Seq_Acc_Cal")

    total_seen = 0
    print_flag = True

    # Prepare CSV file if saving predictions
    csv_data = []
    if save_predictions:
        # Create CSV filename based on model checkpoint
        checkpoint_name = os.path.basename(predictor.checkpoint).replace('.pth', '')
        csv_filename = f"predictions_{checkpoint_name}_{test_size if not on_all else 'all'}_samples.csv"
        csv_path = os.path.join(config.root_dir, csv_filename)

    for batch in pbar:
        original_batch, predicted_batch = predictor.predict_batch(batch, vocab, raw_tokens=True)

        for i in range(len(original_batch)):
            original_tokens = original_batch[i].detach().cpu().numpy().tolist()
            predicted_tokens = predicted_batch[i].detach().cpu().numpy().tolist()

            original = decode_sequence(original_tokens, vocab)
            predicted = decode_sequence(predicted_tokens, vocab)

            assert original != 'Thestringistoolong.' or predicted != 'Thestringistoolong', 'Well now you know where you are getting accuracy from'

            # Check if prediction matches (using original logic)
            is_correct = original == predicted[:min(len(predicted), len(original))]

            if is_correct:
                count += 1

            # Store data for CSV
            if save_predictions:
                csv_data.append({
                    'sample_id': total_seen,
                    'target': original,
                    'prediction': predicted,
                    'is_correct': is_correct,
                    'target_length': len(original),
                    'prediction_length': len(predicted)
                })

            total_seen += 1

        print_flag = False
        pbar.set_postfix(seq_accuracy=count / total_seen)

    accuracy = count / total_seen
    print(f'Batch accuracy: {accuracy:.4f}')

    # Save to CSV if requested
    if save_predictions and csv_data:
        print(f"Saving predictions to: {csv_path}")

        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['sample_id', 'target', 'prediction', 'is_correct', 'target_length', 'prediction_length']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            writer.writerows(csv_data)

        # Print some statistics
        correct_count = sum(1 for row in csv_data if row['is_correct'])
        total_count = len(csv_data)

        print(f"CSV saved with {total_count} samples")
        print(f"Correct predictions: {correct_count}/{total_count} ({correct_count/total_count:.4f})")

        # Additional statistics
        avg_target_len = sum(row['target_length'] for row in csv_data) / total_count
        avg_pred_len = sum(row['prediction_length'] for row in csv_data) / total_count

        print(f"Average target length: {avg_target_len:.2f}")
        print(f"Average prediction length: {avg_pred_len:.2f}")

    return accuracy