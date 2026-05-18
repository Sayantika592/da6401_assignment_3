# DA6401 Assignment 3 — Attention Is All You Need

Implementation of a Transformer-based Neural Machine Translation system from scratch in PyTorch based on the paper:

> **Attention Is All You Need — Vaswani et al., 2017**

This project implements:
- Multi-Head Self Attention
- Encoder-Decoder Transformer
- Noam Learning Rate Scheduler
- Label Smoothing
- Attention Visualization
- BLEU Evaluation
- Positional Encoding Ablations
- W&B Experiment Tracking

The model is trained on the **Multi30k German → English Translation Dataset**.

---

# GitHub Repository

https://github.com/Sayantika592/da6401_assignment_3

---

# W&B Report
https://api.wandb.ai/links/me22b190-indian-institute-of-technology-madras/9mn4rlm3

---

# Project Structure

```bash
├── dataset.py          # Multi30k dataset loading and tokenization
├── lr_scheduler.py     # Noam learning rate scheduler
├── model.py            # Transformer architecture implementation
├── train.py            # Training pipeline and BLEU evaluation
├── checkpoint.pt
├── best_checkpoint.pt
└── README.md
