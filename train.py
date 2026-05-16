"""
train.py — Training Pipeline, Inference & Evaluation
DA6401 Assignment 3: "Attention Is All You Need"

AUTOGRADER CONTRACT (DO NOT MODIFY SIGNATURES):
  ┌─────────────────────────────────────────────────────────────────────┐
  │  greedy_decode(model, src, src_mask, max_len, start_symbol)         │
  │      → torch.Tensor  shape [1, out_len]  (token indices)            │
  │                                                                     │
  │  evaluate_bleu(model, test_dataloader, tgt_vocab, device)           │
  │      → float  (corpus-level BLEU score, 0–100)                      │
  │                                                                     │
  │  save_checkpoint(model, optimizer, scheduler, epoch, path) → None   │
  │  load_checkpoint(path, model, optimizer, scheduler)        → int    │
  └─────────────────────────────────────────────────────────────────────┘
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Optional
from tqdm import tqdm

from model import Transformer, make_src_mask, make_tgt_mask


# ══════════════════════════════════════════════════════════════════════
#  LABEL SMOOTHING LOSS  
# ══════════════════════════════════════════════════════════════════════

class LabelSmoothingLoss(nn.Module):
    """
    Label smoothing as in "Attention Is All You Need"

    Smoothed target distribution:
        y_smooth = (1 - eps) * one_hot(y) + eps / (vocab_size - 1)

    Args:
        vocab_size (int)  : Number of output classes.
        pad_idx    (int)  : Index of <pad> token — receives 0 probability.
        smoothing  (float): Smoothing factor ε (default 0.1).
    """

    def __init__(self, vocab_size: int, pad_idx: int, smoothing: float = 0.1) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits : shape [batch * tgt_len, vocab_size]  (raw model output)
            target : shape [batch * tgt_len]              (gold token indices)

        Returns:
            Scalar loss value.
        """
        assert logits.dim() == 2 and target.dim() == 1

        log_probs = F.log_softmax(logits, dim=-1)  # (N, vocab_size)

        # NLL component: - log_prob of correct token
        nll_loss = -log_probs.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)

        # Smooth component: - mean of all log_probs
        smooth_loss = -log_probs.sum(dim=-1) / self.vocab_size

        # Combined loss
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss

        # Mask out padding positions
        non_pad_mask = target.ne(self.pad_idx)
        loss = loss.masked_select(non_pad_mask).mean()

        return loss


# ══════════════════════════════════════════════════════════════════════
#   TRAINING LOOP  
# ══════════════════════════════════════════════════════════════════════

def run_epoch(
    data_iter,
    model: Transformer,
    loss_fn: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler=None,
    epoch_num: int = 0,
    is_train: bool = True,
    device: str = "cpu",
    pad_idx: int = 1,
) -> float:
    """
    Run one epoch of training or evaluation.

    Args:
        data_iter  : DataLoader yielding (src, tgt) batches of token indices.
        model      : Transformer instance.
        loss_fn    : LabelSmoothingLoss (or any nn.Module loss).
        optimizer  : Optimizer (None during eval).
        scheduler  : NoamScheduler instance (None during eval).
        epoch_num  : Current epoch index (for logging).
        is_train   : If True, perform backward pass and scheduler step.
        device     : 'cpu' or 'cuda'.
        pad_idx    : Padding token index.

    Returns:
        avg_loss : Average loss over the epoch (float).

    """
    if is_train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_tokens = 0

    pbar = tqdm(data_iter, desc=f"{'Train' if is_train else 'Val'} Epoch {epoch_num}", leave=False)

    for src, tgt in pbar:
        src, tgt = src.to(device), tgt.to(device)

        # Teacher forcing: feed tgt[:-1], predict tgt[1:]
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        # Build masks
        src_mask = make_src_mask(src, pad_idx)
        tgt_mask = make_tgt_mask(tgt_input, pad_idx)

        # Forward
        if is_train:
            logits = model(src, tgt_input, src_mask, tgt_mask)
        else:
            with torch.no_grad():
                logits = model(src, tgt_input, src_mask, tgt_mask)

        # Reshape for loss: (batch*tgt_len, vocab_size) and (batch*tgt_len,)
        logits_flat = logits.contiguous().view(-1, logits.size(-1))
        target_flat = tgt_output.contiguous().view(-1)

        loss = loss_fn(logits_flat, target_flat)

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

        num_tokens = tgt_output.ne(pad_idx).sum().item()
        total_loss += loss.item() * num_tokens
        total_tokens += num_tokens

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / max(total_tokens, 1)


# ══════════════════════════════════════════════════════════════════════
#   GREEDY DECODING  
# ══════════════════════════════════════════════════════════════════════

def greedy_decode(
    model: Transformer,
    src: torch.Tensor,
    src_mask: torch.Tensor,
    max_len: int,
    start_symbol: int,
    end_symbol: int,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Generate a translation token-by-token using greedy decoding.

    Args:
        model        : Trained Transformer.
        src          : Source token indices, shape [1, src_len].
        src_mask     : shape [1, 1, 1, src_len].
        max_len      : Maximum number of tokens to generate.
        start_symbol : Vocabulary index of <sos>.
        end_symbol   : Vocabulary index of <eos>.
        device       : 'cpu' or 'cuda'.

    Returns:
        ys : Generated token indices, shape [1, out_len].
             Includes start_symbol; stops at (and includes) end_symbol
             or when max_len is reached.

    """
    model.eval()
    src = src.to(device)
    src_mask = src_mask.to(device)

    # Encode source once
    with torch.no_grad():
        memory = model.encode(src, src_mask)

    # Start with SOS token
    ys = torch.tensor([[start_symbol]], dtype=torch.long, device=device)

    pad_idx = 1  # default pad index

    for _ in range(max_len - 1):
        tgt_mask = make_tgt_mask(ys, pad_idx).to(device)
        with torch.no_grad():
            logits = model.decode(memory, src_mask, ys, tgt_mask)
        # Greedy: pick highest probability token at last position
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        ys = torch.cat([ys, next_token], dim=1)

        if next_token.item() == end_symbol:
            break

    return ys


# ══════════════════════════════════════════════════════════════════════
#   BLEU EVALUATION  
# ══════════════════════════════════════════════════════════════════════

def evaluate_bleu(
    model: Transformer,
    test_dataloader: DataLoader,
    tgt_vocab,
    device: str = "cpu",
    max_len: int = 100,
) -> float:
    """
    Evaluate translation quality with corpus-level BLEU score.

    Args:
        model           : Trained Transformer (in eval mode).
        test_dataloader : DataLoader over the test split.
                          Each batch yields (src, tgt) token-index tensors.
        tgt_vocab       : Vocabulary object with idx_to_token mapping.
                          Must support  tgt_vocab.itos[idx]  or
                          tgt_vocab.lookup_token(idx).
        device          : 'cpu' or 'cuda'.
        max_len         : Max decode length per sentence.

    Returns:
        bleu_score : Corpus-level BLEU (float, range 0–100).

    """
    import evaluate as eval_lib

    bleu_metric = eval_lib.load("bleu")

    model.eval()
    predictions = []
    references = []

    pad_idx = 1
    sos_idx = 2
    eos_idx = 3

    # Build idx→token mapping
    if isinstance(tgt_vocab, dict):
        # tgt_vocab is token→idx, need to invert
        idx_to_token = {v: k for k, v in tgt_vocab.items()}
    else:
        # Assume it has an itos attribute or is already idx→token
        idx_to_token = tgt_vocab.itos if hasattr(tgt_vocab, 'itos') else tgt_vocab

    special_tokens = {"<pad>", "<sos>", "<bos>", "<eos>", "<unk>"}

    for src, tgt in test_dataloader:
        src = src.to(device)

        # Process each sentence in the batch individually
        for i in range(src.size(0)):
            src_i = src[i].unsqueeze(0)  # (1, src_len)
            src_mask = make_src_mask(src_i, pad_idx).to(device)

            # Greedy decode
            pred_tokens = greedy_decode(
                model, src_i, src_mask, max_len,
                start_symbol=sos_idx, end_symbol=eos_idx, device=device,
            )

            # Convert prediction to words
            pred_indices = pred_tokens[0].cpu().tolist()
            pred_words = []
            for idx in pred_indices[1:]:  # skip SOS
                if idx == eos_idx:
                    break
                token = idx_to_token.get(idx, "<unk>")
                if token not in special_tokens:
                    pred_words.append(token)

            # Convert reference to words
            ref_indices = tgt[i].cpu().tolist()
            ref_words = []
            for idx in ref_indices[1:]:  # skip SOS
                if idx == eos_idx:
                    break
                if idx != pad_idx:
                    token = idx_to_token.get(idx, "<unk>")
                    if token not in special_tokens:
                        ref_words.append(token)

            predictions.append(" ".join(pred_words))
            references.append([" ".join(ref_words)])

    results = bleu_metric.compute(predictions=predictions, references=references)
    return results["bleu"] * 100  # range 0–100


# ══════════════════════════════════════════════════════════════════════
# ❺  CHECKPOINT UTILITIES  (autograder loads your model from disk)
# ══════════════════════════════════════════════════════════════════════

def save_checkpoint(
    model: Transformer,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    path: str = "checkpoint.pt",
) -> None:
    """
    Save model + optimiser + scheduler state to disk.

    The autograder will call load_checkpoint to restore your model.
    Do NOT change the keys in the saved dict.

    Args:
        model     : Transformer instance.
        optimizer : Optimizer instance.
        scheduler : NoamScheduler instance.
        epoch     : Current epoch number.
        path      : File path to save to (default 'checkpoint.pt').

    Saves a dict with keys:
        'epoch', 'model_state_dict', 'optimizer_state_dict',
        'scheduler_state_dict', 'model_config'

    model_config must contain all kwargs needed to reconstruct
    Transformer(**model_config), e.g.:
        {'src_vocab_size': ..., 'tgt_vocab_size': ...,
         'd_model': ..., 'N': ..., 'num_heads': ...,
         'd_ff': ..., 'dropout': ...}
    """
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'model_config': {
            'd_model': model.d_model,
            'N': model.N,
            'num_heads': model.num_heads,
            'd_ff': model.d_ff,
            'dropout': model.dropout_rate,
            'load_weights': False,
        },
    }, path)


def load_checkpoint(
    path: str,
    model: Transformer,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
) -> int:
    """
    Restore model (and optionally optimizer/scheduler) state from disk.

    Args:
        path      : Path to checkpoint file saved by save_checkpoint.
        model     : Uninitialised Transformer with matching architecture.
        optimizer : Optimizer to restore (pass None to skip).
        scheduler : Scheduler to restore (pass None to skip).

    Returns:
        epoch : The epoch at which the checkpoint was saved (int).

    """
    checkpoint = torch.load(path, map_location="cpu")

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    return checkpoint['epoch']


# ══════════════════════════════════════════════════════════════════════
#   EXPERIMENT ENTRY POINT
# ══════════════════════════════════════════════════════════════════════

def run_training_experiment() -> None:
    """
    Set up and run the full training experiment.

    Steps:
        1. Init W&B:   wandb.init(project="da6401-a3", config={...})
        2. Build dataset / vocabs from dataset.py
        3. Create DataLoaders for train / val splits
        4. Instantiate Transformer with hyperparameters from config
        5. Instantiate Adam optimizer (β1=0.9, β2=0.98, ε=1e-9)
        6. Instantiate NoamScheduler(optimizer, d_model, warmup_steps=4000)
        7. Instantiate LabelSmoothingLoss(vocab_size, pad_idx, smoothing=0.1)
        8. Training loop:
               for epoch in range(num_epochs):
                   run_epoch(train_loader, model, loss_fn,
                             optimizer, scheduler, epoch, is_train=True)
                   run_epoch(val_loader, model, loss_fn,
                             None, None, epoch, is_train=False)
                   save_checkpoint(model, optimizer, scheduler, epoch)
        9. Final BLEU on test set:
               bleu = evaluate_bleu(model, test_loader, tgt_vocab)
               wandb.log({'test_bleu': bleu})
    """
    import wandb
    from dataset import prepare_data, Multi30kDataset
    from lr_scheduler import NoamScheduler

    # ── Hyperparameters ──
    config = {
        "d_model": 256,
        "N": 3,
        "num_heads": 8,
        "d_ff": 1024,
        "dropout": 0.1,
        "batch_size": 128,
        "num_epochs": 30,
        "warmup_steps": 4000,
        "label_smoothing": 0.1,
        "freq_threshold": 2,
    }

    # 1. Init W&B
    wandb.init(project="da6401-a3", config=config)
    config = wandb.config

    # 2-3. Build dataset / vocabs / DataLoaders
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader, train_ds = prepare_data(
        batch_size=config.batch_size,
        freq_threshold=config.freq_threshold,
    )

    src_vocab = train_ds.src_vocab
    tgt_vocab = train_ds.tgt_vocab
    pad_idx = Multi30kDataset.PAD_IDX
    src_vocab_size = len(src_vocab)
    tgt_vocab_size = len(tgt_vocab)

    # 4. Instantiate Transformer
    model = Transformer(
        d_model=config.d_model,
        N=config.N,
        num_heads=config.num_heads,
        d_ff=config.d_ff,
        dropout=config.dropout,
        freq_threshold=config.freq_threshold,
        load_weights=False,
    ).to(device)

    # Attach vocabs for infer()
    model.src_vocab = src_vocab
    model.tgt_vocab = tgt_vocab
    model.pad_idx = pad_idx

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")

    # 5. Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1.0,  # base LR; actual LR controlled by scheduler
        betas=(0.9, 0.98),
        eps=1e-9,
    )

    # 6. Scheduler
    scheduler = NoamScheduler(optimizer, d_model=config.d_model, warmup_steps=config.warmup_steps)

    # 7. Loss
    loss_fn = LabelSmoothingLoss(
        vocab_size=tgt_vocab_size,
        pad_idx=pad_idx,
        smoothing=config.label_smoothing,
    )

    # 8. Training loop
    best_val_loss = float('inf')

    for epoch in range(config.num_epochs):
        train_loss = run_epoch(
            train_loader, model, loss_fn,
            optimizer, scheduler, epoch, is_train=True, device=device, pad_idx=pad_idx,
        )

        val_loss = run_epoch(
            val_loader, model, loss_fn,
            None, None, epoch, is_train=False, device=device, pad_idx=pad_idx,
        )

        print(f"Epoch {epoch+1}/{config.num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "learning_rate": optimizer.param_groups[0]["lr"],
        })

        # Save checkpoint
        save_checkpoint(model, optimizer, scheduler, epoch, path="checkpoint.pt")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, scheduler, epoch, path="best_checkpoint.pt")
            print(f"  -> Saved best checkpoint (val_loss={val_loss:.4f})")

    # 9. Final BLEU on test set
    print("\nLoading best checkpoint for test evaluation...")
    load_checkpoint("best_checkpoint.pt", model)
    bleu = evaluate_bleu(model, test_loader, tgt_vocab, device=device)
    print(f"Test BLEU: {bleu:.2f}")
    wandb.log({"test_bleu": bleu})

    wandb.finish()


if __name__ == "__main__":
    run_training_experiment()
