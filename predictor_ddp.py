import os

import torch
from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel as DDP

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
        use_ddp (bool, optional): Whether to use DistributedDataParallel. Defaults to True.

    Attributes:
        model (Model): Trained model for prediction.
        ddp_model (Model): DDP-wrapped model (if use_ddp=True).
        path (str): Path to the trained model checkpoint.
        device (str or int): Device for inference.
        checkpoint (str): Model checkpoint filename.
        max_len (int): Maximum target sequence length for inference.
        use_ddp (bool): Whether DDP is being used.
        local_rank (int): Local rank for distributed setup.
        is_master (bool): Whether this is the master process.
    """

    def __init__(self, config, load_best=True, epoch=None, use_ddp=True):
        self.use_ddp = use_ddp
        
        # Set up distributed environment if using DDP
        if self.use_ddp:
            self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
            self.global_rank = int(os.environ.get("RANK", 0))
            self.device = self.local_rank
            self.is_master = self.local_rank == 0
        else:
            self.device = config.device if hasattr(config, 'device') else 0
            self.is_master = True
        
        # Initialize model
        self.model = get_model(config)
        self.model.to(self.device)
        
        # Wrap with DDP if requested
        if self.use_ddp:
            self.ddp_model = DDP(self.model, device_ids=[self.device], find_unused_parameters=True)
        else:
            self.ddp_model = self.model
        
        # Set up checkpoint path
        self.checkpoint = (
            f"{config.model_name}/best.pth"
            if load_best else f"{config.model_name}/ep{epoch + 1}.pth"
        )
        self.path = os.path.join(config.root_dir, self.checkpoint)
        
        # Set dtype
        if config.dtype == 'bfloat16':
            self.dtype = torch.bfloat16
        elif config.dtype == 'float32':
            self.dtype = torch.float32
        elif config.dtype == 'float16':
            self.dtype = torch.float16

        # Load model checkpoint
        state = torch.load(self.path, map_location=f"cuda:{self.device}")
        
        # Load state dict into the base model (not DDP wrapper)
        self.model.load_state_dict(state['state_dict'])
        
        self.max_len = config.tgt_max_len

        if self.is_master:
            print(f"Using epoch {state['epoch']} model for predictions.")

    def predict(self, test_example, vocab, raw_tokens=False):
        """
        Generates predictions for a given test example.

        Args:
            test_example (tuple): Tuple containing source tensor and original tokens.
            vocab (Vocab): Vocabulary object with decode method.
            raw_tokens (bool, optional): Whether to return raw token outputs. Defaults to False.

        Returns:
            str or tuple: Decoded equation or tuple of original and generated tokens.
        """
        # Use the base model for inference (not DDP wrapper)
        self.model.eval()

        src = test_example[0].unsqueeze(0).to(self.device)
        src_padding_mask, _ = create_mask(
            src, torch.zeros((1, 1), dtype=self.dtype, device=self.device)
        )

        tgt_tokens = greedy_decode(
            self.model, self.device, self.max_len,
            src, src_padding_mask, test_example[1][0], self.dtype
        ).flatten()

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
        # Use the base model for inference (not DDP wrapper)
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

        # Use base model for greedy decode
        tgt_tokens_batch = greedy_decode(
            self.model,
            self.device,
            self.max_len,
            src_batch,
            src_padding_mask,
            bos_tokens[0],
            self.dtype
        )

        if raw_tokens:
            return original_tokens_batch, tgt_tokens_batch

        decoded_predictions = [''.join(vocab.decode(pred)) for pred in tgt_tokens_batch]
        return decoded_predictions


def sequence_accuracy(config, test_ds, vocab, load_best=True, epoch=None, on_all=False, test_size=300, batch_size=16):
    """
    Calculate the sequence accuracy with DDP support.

    Args:
        config (object): Configuration for inference.
        test_ds (Dataset): Dataset for testing.
        vocab (Vocab): Vocabulary object.
        load_best (bool, optional): Whether to load the best model. Defaults to True.
        epoch (int, optional): Specific epoch to load. Defaults to None.
        on_all (bool, optional): Whether to test on all data. Defaults to False.
        test_size (int, optional): Number of test samples to evaluate. Defaults to 300.
        batch_size (int, optional): Batch size for evaluation. Defaults to 16.

    Returns:
        float: Sequence accuracy.
    """
    # Check if we're in a distributed environment
    use_ddp = "LOCAL_RANK" in os.environ and "RANK" in os.environ
    
    predictor = Predictor(config, load_best, epoch, use_ddp=use_ddp)
    count = 0
    num_samples = 10 if config.debug else test_size
    num_samples = min(num_samples, len(test_ds))
    
    if on_all:
        sub_dataset = test_ds
        total_samples = len(test_ds)
    else:
        random_idx = generate_unique_random_integers(
            num_samples, start=0, end=len(test_ds)
        )
        sub_dataset = Subset(test_ds, random_idx)
        total_samples = num_samples

    # Create distributed sampler if using DDP
    if use_ddp:
        sampler = torch.utils.data.DistributedSampler(
            sub_dataset,
            num_replicas=int(os.environ.get("WORLD_SIZE", 1)),
            rank=int(os.environ.get("RANK", 0)),
            shuffle=False
        )
        dataloader = DataLoader(
            sub_dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=config.num_workers,
            pin_memory=True,
            collate_fn=collate_fn
        )
    else:
        dataloader = DataLoader(
            sub_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True,
            collate_fn=collate_fn
        )

    # Only show progress bar on master process
    pbar = tqdm(dataloader, disable=not predictor.is_master)
    pbar.set_description("Seq_Acc_Cal")

    total_seen = 0
    local_count = 0
    local_total = 0
    
    with torch.no_grad():
        for batch in pbar:
            original_batch, predicted_batch = predictor.predict_batch(batch, vocab, raw_tokens=True)

            for i in range(len(original_batch)):
                original_tokens = original_batch[i].detach().cpu().numpy().tolist()
                predicted_tokens = predicted_batch[i].detach().cpu().numpy().tolist()

                original = decode_sequence(original_tokens, vocab)
                predicted = decode_sequence(predicted_tokens, vocab)

                assert original != 'Thestringistoolong.' or predicted != 'Thestringistoolong', \
                    'Well now you know where you are getting accuracy from'

                if original == predicted[:min(len(predicted), len(original))]:
                    local_count += 1
                local_total += 1
            
            if predictor.is_master:
                pbar.set_postfix(seq_accuracy=local_count / local_total)

    # Aggregate results across all processes if using DDP
    if use_ddp:
        # Convert to tensors for all_reduce
        local_count_tensor = torch.tensor(local_count, dtype=torch.long, device=predictor.device)
        local_total_tensor = torch.tensor(local_total, dtype=torch.long, device=predictor.device)
        
        # Sum across all processes
        torch.distributed.all_reduce(local_count_tensor, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(local_total_tensor, op=torch.distributed.ReduceOp.SUM)
        
        global_accuracy = local_count_tensor.item() / local_total_tensor.item()
    else:
        global_accuracy = local_count / local_total

    if predictor.is_master:
        print(f'Final accuracy: {global_accuracy:.4f}')
    
    return global_accuracy