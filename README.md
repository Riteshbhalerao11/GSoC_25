# SYMBA

SYMBA is a research codebase for sequence-to-sequence modeling of symbolic scattering amplitudes. 

## Highlights
- FlashAttention-based encoder–decoder with optional SineKAN layers and mixed-precision support.
- Custom tokenizer/vocabulary for amplitude (`amp`) and squared amplitude (`sqamp`) expressions.
- Distributed training with PyTorch `torchrun`, gradient scaling, checkpoint rotation, and Weights & Biases logging.
- Self-critical sequence training (SCST) fine-tuning with edit-distance rewards and temperature-controlled sampling (experimental, lightly tested).
- Greedy and beam-search decoders for batched evaluation and offline prediction (beam search currently unverified).

## Data Layout
Training expects CSV files with at least two columns:

| column | description                           |
| ------ | ------------------------------------- |
| `amp`  | symbolic amplitude expression (input) |
| `sqamp`| squared amplitude (target)            |

Place `train.csv`, `valid.csv`, and `test.csv` inside the directory passed as `--data_dir`. Tokenization relies on these column names and may raise if sequences exceed configured `src_max_len` / `tgt_max_len`.

## Training
Launch distributed training with `torchrun` (single node example with 2 GPUs):

```bash
torchrun --standalone --nproc_per_node 2 -m Flash.main \
  --project_name Vanilla_EW \
  --run_name run_001 \
  --model_name vanilla_flash \
  --root_dir /path/to/checkpoints \
  --data_dir /path/to/data/ \
  --device cuda \
  --epochs 200 \
  --training_batch_size 64 \
  --valid_batch_size 64 \
  --num_workers 32 \
  --embedding_size 512 \
  --ff_dims 4096 \
  --nhead 8 \
  --num_encoder_layers 3 \
  --num_decoder_layers 3 \
  --warmup_ratio 0.05 \
  --dropout 0.1 \
  --weight_decay 0.01 \
  --optimizer_lr 5e-5 \
  --src_max_len 810 \
  --tgt_max_len 1730 \
  --curr_epoch 0 \
  --world_size 2 \
  --save_freq 10 \
  --log_freq 20 \
  --train_shuffle \
  --pin_memory
```

All CLI flags are defined in `Flash/fn_utils.py::parse_args`. Adjust `--is_kan`, `--kan_ff_dims`, `--is_pre_norm`, `--is_constant_lr`, etc. to toggle architectural and optimization variants. Enable WandB logging with `wandb login` or set the `WANDB_API_KEY` environment variable before launching.

### Running on SLURM
Sample batch scripts live under `run_experiments/`. They load the site PyTorch module, reserve GPUs, and invoke `torchrun`. Adapt account, queue, and filesystem paths before submission.

## Fine-Tuning (SCST)
The SCST path is highly experimental and has only undergone limited runs; expect rough edges and monitor metrics closely. To continue from a trained checkpoint:
- Train a base model with `--finetune` unset.
- Place indices of inconsistent samples in `root_dir/finetune/inc_idx.npy` (as produced by prior analysis).
- Re-launch `Flash.main` with `--finetune` and temperature/top-k sampling options; the SCST pipeline will sample corrective data, compute edit-distance rewards, and write results to `root_dir/finetune`.

## Evaluation & Inference
`Flash.predictor.Predictor` loads checkpoints, performs greedy decoding, and can optionally invoke beam search (the beam-search branch is not yet validated in production). Integrate it inside notebooks or scripts for batch scoring:

```python
from Flash.predictor import Predictor, sequence_accuracy
from Flash.config import ModelTestConfig

config = ModelTestConfig(
    model_name="vanilla_flash",
    root_dir="/path/to/checkpoints",
    data_dir="/path/to/data/",
    device="cuda",
    embedding_size=512,
    nhead=8,
    num_encoder_layers=3,
    num_decoder_layers=3,
    ff_dims=4096,
    dropout=0.1,
    src_max_len=810,
    tgt_max_len=1730,
    test_batch_size=32,
    kan_ff_dims=[4096, 4096],
    is_kan=False,
    is_pre_norm=False,
    use_torch_mha=False,
    is_termwise=True,
    is_beamsearch=False,
)
```

Use `sequence_accuracy(config, valid_ds, vocab)` to compute exact-match rates. The tokenizer (`Flash/tokenizer.py`) enables reproducible preprocessing for both training and evaluation splits.

## Repository Structure
- `main.py` — entry-point orchestrating config parsing, tokenizer creation, and Trainer/SCSTrainer selection.
- `trainer.py` / `finetune.py` — distributed training loops for MLE and SCST.
- `model/` — FlashAttention-enhanced Transformer, optional SineKAN components, embeddings, and utilities.
- `tokenizer.py` / `data.py` — domain-specific tokenization and dataset objects.
- `predictor.py` / `inference.py` — decoding utilities and accuracy evaluation helpers.
- `run_experiments/` — SLURM launcher templates for EW/QED/QCD datasets.

## Logging & Checkpoints
Checkpoints are written into `root_dir` with epoch suffixes and a `*_best.pth` symlink. Metrics, gradients, and learning-rate schedules stream to Weights & Biases; set `--save_limit` to control retention and `--resume_best` / `--run_id` to continue interrupted jobs.

---

Flash is under active development; contributions and experiment reports are welcome. Open an issue or reach out before large refactors to keep training scripts aligned with current HPC requirements.

