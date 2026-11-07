- **Model**: Pre-norm encoder-decoder Transformer implemented from scratch in hw3lib (26.5M parameters).
  Conv2D speech embedding with 4x time reduction, 12 encoder layers (bidirectional
  self-attention) and 6 decoder layers (causal self-attention + cross-attention); d_model 256,
  4 heads, swiglu feed-forward (d_ff 1024), rmsnorm,
  rope positional encoding, QK-norm True, weight tying True, dropout 0.1.
  10k BPE tokens; auxiliary CTC head on the encoder.
- **Training Strategy**: cold_start training for 35 epochs. Loss = cross-entropy (label smoothing
  0.1) + 0.3 x CTC. adamw (peak lr
  0.001, weight decay 0.01), 4-epoch linear warmup then
  cosine decay, batch 32 x 2 gradient accumulation, gradient
  clipping at 1.0, fp16 mixed precision on a T4. Checkpoint and beam width (final: 5)
  selected on dev-clean.
- **Augmentations**: SpecAugment on training batches (2 frequency masks up to 20 bins, 4 time masks up to 50 frames); global mean-variance feature normalization.
- **Notebook Execution**: Clone the repo, add GITHUB_TOKEN / KAGGLE_USERNAME / KAGGLE_API_TOKEN / WANDB_API_KEY as secrets,
  then run all cells top to bottom on a T4 GPU. All settings come from the config.yaml cell.