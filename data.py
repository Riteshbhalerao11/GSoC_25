from torch.utils.data import Dataset
import torch

from .constants import BOS_IDX, PAD_IDX, EOS_IDX, SEP_IDX
from .logger import get_logger

import numpy as np
from tqdm import tqdm
logger = get_logger(__name__)
class Data(Dataset):
    """
    Custom PyTorch dataset for handling data.

    Args:
        df (DataFrame): DataFrame containing data.
    """

    def __init__(self, df, tokenizer, config, src_vocab, tgt_vocab):
        super(Data, self).__init__()
        self.config = config
        self.tgt_tokenize = tokenizer.tgt_tokenize
        self.src_tokenize = tokenizer.src_tokenize
        self.bos_token = torch.tensor([BOS_IDX], dtype=torch.int64)
        self.eos_token = torch.tensor([EOS_IDX], dtype=torch.int64)
        self.pad_token = torch.tensor([PAD_IDX], dtype=torch.int64)
        self.sep_token = torch.tensor([SEP_IDX], dtype=torch.int64)
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        # self.test=test


        # if self.config.filter_len:
        #     logger.info(f'Starting Data Filtering')
        #     df_new = df[
        #         (df['sqamp'].apply(self.tgt_tokenize).str.len() <= self.config.tgt_max_len - 3) &
        #         (df['amp'].apply(self.src_tokenize).str.len() <= self.config.src_max_len - 3)
        #     ].reset_index(drop=True)

        #     assert df_new['sqamp'].apply(self.tgt_tokenize).str.len().max() <= self.config.tgt_max_len - 3, 'Problem with SqAmp'
        #     assert df_new['amp'].apply(self.src_tokenize).str.len().max() <= self.config.src_max_len - 3, 'Problem with Amp'

        #     logger.info(f"Filtered data size is: {len(df_new)} -> {1.0 - (len(df_new) / len(df))}")
        #     logger.info(f"New Max amp Sequence length: {df['amp'].str.len().max()}")
        #     logger.info(f"New Max sqamp Sequence length: {df['sqamp'].str.len().max()}")
        # else:
        #     df_new = df
        if self.config.filter_len:
            logger.info(f'Starting Data Filtering')
            # Enable tqdm for pandas operations
            tqdm.pandas()

            # Pre-calculate lengths in batches to avoid repeated tokenization
            logger.info("Calculating sequence lengths...")

            # Use vectorized string length first as a pre-filter (much faster)
            sqamp_char_lens = df['sqamp'].str.len()
            amp_char_lens = df['amp'].str.len()
            # Rough estimate: assume ~4 chars per token on average for pre-filtering
            # This eliminates obviously too-long sequences before expensive tokenization
            # char_len_filter = (
            #     (sqamp_char_lens <= (self.config.tgt_max_len - 3) * 6) &  # Conservative multiplier
            #     (amp_char_lens <= (self.config.src_max_len - 3) * 6)
            # )

            # logger.info(f"Pre-filter removed {(~char_len_filter).sum()} rows based on character length")
            # df_prefiltered = df[char_len_filter].reset_index(drop=True)
            df_prefiltered = df

            if len(df_prefiltered) == 0:
                logger.warning("All rows filtered out by character length pre-filter")
                return df_prefiltered

            # Batch tokenization for remaining rows
            batch_size = 10000
            sqamp_lens = []
            amp_lens = []

            logger.info("Tokenizing and calculating exact lengths...")
            for i in tqdm(range(0, len(df_prefiltered), batch_size), desc="Processing batches"):
                batch_end = min(i + batch_size, len(df_prefiltered))
                batch_df = df_prefiltered.iloc[i:batch_end]

                # Tokenize batch
                sqamp_batch_lens = batch_df['sqamp'].apply(self.tgt_tokenize).str.len()
                amp_batch_lens = batch_df['amp'].apply(self.src_tokenize).str.len()

                sqamp_lens.extend(sqamp_batch_lens.tolist())
                amp_lens.extend(amp_batch_lens.tolist())

            # Convert to numpy arrays for faster operations
            sqamp_lens = np.array(sqamp_lens)
            amp_lens = np.array(amp_lens)

            # Apply final filter
            final_filter = (
                (sqamp_lens <= self.config.tgt_max_len - 3) &
                (amp_lens <= self.config.src_max_len - 3)
            )

            df_new = df_prefiltered[final_filter].reset_index(drop=True)

            # Assertions using pre-computed lengths
            filtered_sqamp_lens = sqamp_lens[final_filter]
            filtered_amp_lens = amp_lens[final_filter]

            assert filtered_sqamp_lens.max() <= self.config.tgt_max_len - 3, 'Problem with SqAmp'
            assert filtered_amp_lens.max() <= self.config.src_max_len - 3, 'Problem with Amp'

            logger.info(f"Filtered data size is: {len(df_new)} -> {1.0 - (len(df_new) / len(df)):.4f} removed")
            logger.info(f"New Max amp Sequence length: {df_new['amp'].str.len().max()}")
            logger.info(f"New Max sqamp Sequence length: {df_new['sqamp'].str.len().max()}")
        else:
            df_new = df
        self.tgt_vals = df_new['sqamp']
        self.src_vals = df_new['amp']



    def __len__(self):
        """
        Get the length of the dataset.

        Returns:
            int: Length of the dataset.
        """
        return len(self.src_vals)

    def __getitem__(self, idx):
        """
        Get an item from the dataset at the specified index.

        Args:
            idx (int): Index of the item.

        Returns:
            tuple: Tuple containing source and target tensors.
        """
        src_tokenized = self.src_tokenize(self.src_vals[idx])
        tgt_tokenized = self.tgt_tokenize(self.tgt_vals[idx])

        src_ids = self.src_vocab.encode(src_tokenized)
        tgt_ids = self.tgt_vocab.encode(tgt_tokenized)

        enc_excess_tokens = self.config.src_max_len - len(src_ids) - 3
        dec_excess_tokens = self.config.tgt_max_len - len(tgt_ids) - 3

        if self.config.truncate:
            if enc_excess_tokens < 0:
                src_ids = src_ids[:self.config.src_max_len-3]
            if dec_excess_tokens < 0:
                tgt_ids = tgt_ids[:self.config.tgt_max_len-3]
        else:
            if enc_excess_tokens < 0 or dec_excess_tokens < 0:
                raise ValueError(f"idx: {idx} Sentence is too long \n enc_excess_tokens: {enc_excess_tokens} \n dec_excess_tokens: {dec_excess_tokens}")
        if self.config.lm_head:
            src_tensor = torch.cat(
                [
                    self.bos_token,
                    torch.tensor(src_ids, dtype=torch.int64),
                    self.eos_token,
                    self.pad_token,
                ],
                dim=0,
            )
            tgt_tensor = torch.cat(
                [
                    self.bos_token,
                    torch.tensor(tgt_ids, dtype=torch.int64),
                    self.eos_token,
                    self.pad_token,
                ],
                dim=0,
            )

            input_tensor = torch.cat(
                [
                    self.bos_token,
                    torch.tensor(src_ids, dtype=torch.int64),
                    self.sep_token,
                    torch.tensor(tgt_ids, dtype=torch.int64),
                    self.eos_token,
                    self.pad_token,
                ],
                dim=0,
            )
            return input_tensor, tgt_tensor

        elif self.config.is_termwise:
        # if self.config.is_termwise:
            src_tensor = torch.cat(
                [
                    torch.tensor(src_ids, dtype=torch.int64),
                    self.pad_token,
                ],
                dim=0,
            )
            tgt_tensor = torch.cat(
                [
                    torch.tensor(tgt_ids, dtype=torch.int64),
                    self.pad_token,

                ],
                dim=0,
            )
        else:
            # src_tensor = torch.cat(
            #     [
            #         self.bos_token,
            #         torch.tensor(src_ids, dtype=torch.int64),
            #         self.eos_token,
            #         self.pad_token,
            #     ],
            #     dim=0,
            # )
            # tgt_tensor = torch.cat(
            #     [
            #         self.bos_token,
            #         torch.tensor(tgt_ids, dtype=torch.int64),
            #         self.eos_token,
            #         self.pad_token,

            #     ],
            #     dim=0,
            # )
            src_tensor = torch.cat(
                [
                    self.bos_token,
                    torch.tensor(src_ids, dtype=torch.int64),
                    self.eos_token,
                    self.pad_token,
                ],
                dim=0,
            )
            tgt_tensor = torch.cat(
                [
                    self.bos_token,
                    torch.tensor(tgt_ids, dtype=torch.int64),
                    self.eos_token,
                    self.pad_token,
                ],
                dim=0,
            )
        return src_tensor, tgt_tensor

    @staticmethod
    def get_data(df_train, df_test, df_valid, config, tokenizer, src_vocab,tgt_vocab):
        """
        Create datasets (train, test, and valid)

        Returns:
            dict: Dictionary containing train, test, and valid datasets.
        """
        train = Data(df_train, tokenizer, config,src_vocab,tgt_vocab)
        test = Data(df_test, tokenizer, config,src_vocab,tgt_vocab) if df_test is not None else None
        valid = Data(df_valid, tokenizer, config,src_vocab,tgt_vocab)

        return {'train': train, 'test': test, 'valid': valid}