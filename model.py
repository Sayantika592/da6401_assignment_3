"""
model.py — Transformer Architecture Skeleton
DA6401 Assignment 3: "Attention Is All You Need"

AUTOGRADER CONTRACT (DO NOT MODIFY SIGNATURES):
  ┌─────────────────────────────────────────────────────────────────┐
  │  scaled_dot_product_attention(Q, K, V, mask) → (out, weights)  │
  │  MultiHeadAttention.forward(q, k, v, mask)   → Tensor          │
  │  PositionalEncoding.forward(x)               → Tensor          │
  │  make_src_mask(src, pad_idx)                 → BoolTensor      │
  │  make_tgt_mask(tgt, pad_idx)                 → BoolTensor      │
  │  Transformer.encode(src, src_mask)           → Tensor          │
  │  Transformer.decode(memory,src_m,tgt,tgt_m)  → Tensor          │
  └─────────────────────────────────────────────────────────────────┘
"""

import math
import copy
import os
import gdown
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════════
#   STANDALONE ATTENTION FUNCTION  
#    Exposed at module level so the autograder can import and test it
#    independently of MultiHeadAttention.
# ══════════════════════════════════════════════════════════════════════

def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Scaled Dot-Product Attention.

        Attention(Q, K, V) = softmax( Q·Kᵀ / √dₖ ) · V

    Args:
        Q    : Query tensor,  shape (..., seq_q, d_k)
        K    : Key tensor,    shape (..., seq_k, d_k)
        V    : Value tensor,  shape (..., seq_k, d_v)
        mask : Optional Boolean mask, shape broadcastable to
               (..., seq_q, seq_k).
               Positions where mask is True are MASKED OUT
               (set to -inf before softmax).

    Returns:
        output : Attended output,   shape (..., seq_q, d_v)
        attn_w : Attention weights, shape (..., seq_q, seq_k)
    """
    d_k = Q.size(-1)

    # Q·Kᵀ / √dₖ
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    # Apply mask: True positions → -inf
    if mask is not None:
        scores = scores.masked_fill(mask, float('-inf'))

    attn_w = F.softmax(scores, dim=-1)

    # Replace NaN (from all-masked rows) with 0
    attn_w = attn_w.masked_fill(torch.isnan(attn_w), 0.0)

    output = torch.matmul(attn_w, V)
    return output, attn_w


# ══════════════════════════════════════════════════════════════════════
# ❷  MASK HELPERS 
#    Exposed at module level so they can be tested independently and
#    reused inside Transformer.forward.
# ══════════════════════════════════════════════════════════════════════

def make_src_mask(
    src: torch.Tensor,
    pad_idx: int = 1,
) -> torch.Tensor:
    """
    Build a padding mask for the encoder (source sequence).

    Args:
        src     : Source token-index tensor, shape [batch, src_len]
        pad_idx : Vocabulary index of the <pad> token (default 1)

    Returns:
        Boolean mask, shape [batch, 1, 1, src_len]
        True  → position is a PAD token (will be masked out)
        False → real token
    """
    # True where src == pad_idx (these will be masked out)
    src_mask = (src == pad_idx).unsqueeze(1).unsqueeze(2)
    return src_mask  # (batch, 1, 1, src_len)


def make_tgt_mask(
    tgt: torch.Tensor,
    pad_idx: int = 1,
) -> torch.Tensor:
    """
    Build a combined padding + causal (look-ahead) mask for the decoder.

    Args:
        tgt     : Target token-index tensor, shape [batch, tgt_len]
        pad_idx : Vocabulary index of the <pad> token (default 1)

    Returns:
        Boolean mask, shape [batch, 1, tgt_len, tgt_len]
        True → position is masked out (PAD or future token)
    """
    batch_size, tgt_len = tgt.size()

    # Padding mask: True where pad → (batch, 1, 1, tgt_len)
    tgt_pad_mask = (tgt == pad_idx).unsqueeze(1).unsqueeze(2)

    # Causal (look-ahead) mask: True in upper triangle (future positions)
    # shape: (1, 1, tgt_len, tgt_len)
    tgt_causal_mask = torch.triu(
        torch.ones((tgt_len, tgt_len), device=tgt.device, dtype=torch.bool),
        diagonal=1,
    ).unsqueeze(0).unsqueeze(0)

    # Combine: True if EITHER pad OR future
    tgt_mask = tgt_pad_mask | tgt_causal_mask  # (batch, 1, tgt_len, tgt_len)
    return tgt_mask


# ══════════════════════════════════════════════════════════════════════
#  MULTI-HEAD ATTENTION 
# ══════════════════════════════════════════════════════════════════════

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention as in "Attention Is All You Need", §3.2.2.

        MultiHead(Q,K,V) = Concat(head_1,...,head_h) · W_O
        head_i = Attention(Q·W_Qi, K·W_Ki, V·W_Vi)

    You are NOT allowed to use torch.nn.MultiheadAttention.

    Args:
        d_model   (int)  : Total model dimensionality. Must be divisible by num_heads.
        num_heads (int)  : Number of parallel attention heads h.
        dropout   (float): Dropout probability applied to attention weights.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model   = d_model
        self.num_heads = num_heads
        self.d_k       = d_model // num_heads   # depth per head

        # Linear projections: W_Q, W_K, W_V, W_O
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.attn_dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        query: torch.Tensor,
        key:   torch.Tensor,
        value: torch.Tensor,
        mask:  Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            query : shape [batch, seq_q, d_model]
            key   : shape [batch, seq_k, d_model]
            value : shape [batch, seq_k, d_model]
            mask  : Optional BoolTensor broadcastable to
                    [batch, num_heads, seq_q, seq_k]
                    True → masked out (attend nowhere)

        Returns:
            output : shape [batch, seq_q, d_model]

        """
        batch_size = query.size(0)

        # 1) Linear projections
        Q = self.W_q(query)  # (batch, seq_q, d_model)
        K = self.W_k(key)
        V = self.W_v(value)

        # 2) Reshape: (batch, seq, d_model) → (batch, num_heads, seq, d_k)
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # 3) Scaled dot-product attention
        attn_output, attn_weights = scaled_dot_product_attention(Q, K, V, mask)

        # 4) Apply dropout to attention weights and re-compute
        attn_weights = self.attn_dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, V)

        # 5) Concat heads: (batch, num_heads, seq_q, d_k) → (batch, seq_q, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )

        # 6) Final linear projection W_O
        output = self.W_o(attn_output)
        return output


# ══════════════════════════════════════════════════════════════════════
#   POSITIONAL ENCODING  
# ══════════════════════════════════════════════════════════════════════

class PositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding as in "Attention Is All You Need", §3.5.

    Args:
        d_model  (int)  : Embedding dimensionality.
        dropout  (float): Dropout applied after adding encodings.
        max_len  (int)  : Maximum sequence length to pre-compute (default 5000).
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )  # (d_model/2,)

        pe[:, 0::2] = torch.sin(position * div_term)  # even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # odd indices

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)  # registered as a buffer, NOT a trainable parameter

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : Input embeddings, shape [batch, seq_len, d_model]

        Returns:
            Tensor of same shape [batch, seq_len, d_model]
            = x  +  PE[:, :seq_len, :]  

        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ══════════════════════════════════════════════════════════════════════
#  FEED-FORWARD NETWORK 
# ══════════════════════════════════════════════════════════════════════

class PositionwiseFeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network, §3.3:

        FFN(x) = max(0, x·W₁ + b₁)·W₂ + b₂

    Args:
        d_model (int)  : Input / output dimensionality (e.g. 512).
        d_ff    (int)  : Inner-layer dimensionality (e.g. 2048).
        dropout (float): Dropout applied between the two linears.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : shape [batch, seq_len, d_model]
        Returns:
              shape [batch, seq_len, d_model]
        
        """
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


# ══════════════════════════════════════════════════════════════════════
#  ENCODER LAYER  
# ══════════════════════════════════════════════════════════════════════

class EncoderLayer(nn.Module):
    """
    Single Transformer encoder sub-layer:
        x → [Self-Attention → Add & Norm] → [FFN → Add & Norm]

    Args:
        d_model   (int)  : Model dimensionality.
        num_heads (int)  : Number of attention heads.
        d_ff      (int)  : FFN inner dimensionality.
        dropout   (float): Dropout probability.
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x        : shape [batch, src_len, d_model]
            src_mask : shape [batch, 1, 1, src_len]

        Returns:
            shape [batch, src_len, d_model]

        """
        # Self-attention sub-layer (Post-LayerNorm)
        attn_output = self.self_attn(x, x, x, mask=src_mask)
        x = self.norm1(x + self.dropout1(attn_output))

        # Feed-forward sub-layer
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_output))

        return x


# ══════════════════════════════════════════════════════════════════════
#   DECODER LAYER 
# ══════════════════════════════════════════════════════════════════════

class DecoderLayer(nn.Module):
    """
    Single Transformer decoder sub-layer:
        x → [Masked Self-Attn → Add & Norm]
          → [Cross-Attn(memory) → Add & Norm]
          → [FFN → Add & Norm]

    Args:
        d_model   (int)  : Model dimensionality.
        num_heads (int)  : Number of attention heads.
        d_ff      (int)  : FFN inner dimensionality.
        dropout   (float): Dropout probability.
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(
        self,
        x:        torch.Tensor,
        memory:   torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x        : shape [batch, tgt_len, d_model]
            memory   : Encoder output, shape [batch, src_len, d_model]
            src_mask : shape [batch, 1, 1, src_len]
            tgt_mask : shape [batch, 1, tgt_len, tgt_len]

        Returns:
            shape [batch, tgt_len, d_model]
        """
        # 1) Masked self-attention
        attn_out = self.self_attn(x, x, x, mask=tgt_mask)
        x = self.norm1(x + self.dropout1(attn_out))

        # 2) Cross-attention over encoder memory
        attn_out = self.cross_attn(x, memory, memory, mask=src_mask)
        x = self.norm2(x + self.dropout2(attn_out))

        # 3) Feed-forward
        ff_out = self.feed_forward(x)
        x = self.norm3(x + self.dropout3(ff_out))

        return x


# ══════════════════════════════════════════════════════════════════════
#  ENCODER & DECODER STACKS
# ══════════════════════════════════════════════════════════════════════

class Encoder(nn.Module):
    """Stack of N identical EncoderLayer modules with final LayerNorm."""

    def __init__(self, layer: EncoderLayer, N: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm = nn.LayerNorm(layer.self_attn.d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x    : shape [batch, src_len, d_model]
            mask : shape [batch, 1, 1, src_len]
        Returns:
            shape [batch, src_len, d_model]
        """
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    """Stack of N identical DecoderLayer modules with final LayerNorm."""

    def __init__(self, layer: DecoderLayer, N: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm = nn.LayerNorm(layer.self_attn.d_model)

    def forward(
        self,
        x:        torch.Tensor,
        memory:   torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x        : shape [batch, tgt_len, d_model]
            memory   : shape [batch, src_len, d_model]
            src_mask : shape [batch, 1, 1, src_len]
            tgt_mask : shape [batch, 1, tgt_len, tgt_len]
        Returns:
            shape [batch, tgt_len, d_model]
        """
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


# ══════════════════════════════════════════════════════════════════════
#   FULL TRANSFORMER  
# ══════════════════════════════════════════════════════════════════════

class Transformer(nn.Module):
    """
    Full Encoder-Decoder Transformer for sequence-to-sequence tasks.

    Args:
        src_vocab_size (int)  : Source vocabulary size.
        tgt_vocab_size (int)  : Target vocabulary size.
        d_model        (int)  : Model dimensionality (default 512).
        N              (int)  : Number of encoder/decoder layers (default 6).
        num_heads      (int)  : Number of attention heads (default 8).
        d_ff           (int)  : FFN inner dimensionality (default 2048).
        dropout        (float): Dropout probability (default 0.1).
    """
    GDRIVE_FILE_ID   = "1mt0mm2dRpv0LXAa4Ze702CprfYydQS9e"
    CHECKPOINT_NAME  = "best_model.pt"


    def __init__(
        self,
        d_model:           int   = 256,
        N:                 int   = 3,
        num_heads:         int   = 8,
        d_ff:              int   = 1024,
        dropout:           float = 0.1,
        freq_threshold:    int   = 2,
        use_scaling:       bool  = True,         # [2.2]
        pos_encoding_type: str   = "sinusoidal", # [2.4]
        # Set to False during training to skip gdown + vocab build
        load_weights:      bool  = True,
    ) -> None:
        super().__init__()


        # ── Store config ──
        self.d_model           = d_model
        self.N                 = N
        self.num_heads         = num_heads
        self.d_ff              = d_ff
        self.dropout_rate      = dropout
        self.use_scaling       = use_scaling
        self.pos_encoding_type = pos_encoding_type
        self.pad_idx           = 1   # <pad> is always index 1
 
        # ── Step 1: Load spacy tokenizers (inside __init__) ───────────
        import spacy
        from spacy.cli import download
        try:
            self._spacy_de = spacy.load("de_core_news_sm")
        except OSError:
            download("de_core_news_sm")
            self._spacy_de = spacy.load("de_core_news_sm")
        try:
            self._spacy_en = spacy.load("en_core_web_sm")
        except OSError:
            download("en_core_web_sm")
            self._spacy_en = spacy.load("en_core_web_sm")
 
        # ── Step 2: Build vocab from Multi30k (inside __init__) ───────
        self.src_vocab, self.tgt_vocab = self._build_vocab(freq_threshold)
        self.src_itos = {v: k for k, v in self.src_vocab.items()}
        self.tgt_itos = {v: k for k, v in self.tgt_vocab.items()}
 
        src_vocab_size = len(self.src_vocab)
        tgt_vocab_size = len(self.tgt_vocab)
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
 
        # ── Step 3: Build architecture ────────────────────────────────
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
 
        if pos_encoding_type == "learned":
            self.pos_encoding = LearnedPositionalEncoding(d_model, dropout)
        else:
            self.pos_encoding = PositionalEncoding(d_model, dropout)
 
        encoder_layer      = EncoderLayer(d_model, num_heads, d_ff, dropout,
                                          use_scaling=use_scaling)
        self.encoder       = Encoder(encoder_layer, N)
 
        decoder_layer      = DecoderLayer(d_model, num_heads, d_ff, dropout,
                                          use_scaling=use_scaling)
        self.decoder       = Decoder(decoder_layer, N)
 
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)
        self._init_parameters()
 
        # ── Step 4: Download weights from Drive and load (inside __init__) ──
        if load_weights:
            if not os.path.exists(self.CHECKPOINT_NAME):
                print(f"Downloading weights from Google Drive ({self.GDRIVE_FILE_ID})...")
                gdown.download(
                    id=self.GDRIVE_FILE_ID,
                    output=self.CHECKPOINT_NAME,
                    quiet=False,
                )
            print(f"Loading weights from {self.CHECKPOINT_NAME}...")
            state = torch.load(self.CHECKPOINT_NAME, map_location="cpu")
            self.load_state_dict(state["model_state_dict"])
            print("Weights loaded successfully.")
 
    # ── Vocab builder ──────────────────────────────────────────────────
    def _build_vocab(self, freq_threshold: int = 2):
        """
        Loads Multi30k training split, tokenizes with spacy,
        and builds src (DE) and tgt (EN) vocabularies.
        Special tokens: <unk>=0, <pad>=1, <sos>=2, <eos>=3
        """
        from collections import Counter
        from datasets import load_dataset
 
        SPECIALS  = ["<unk>", "<pad>", "<sos>", "<eos>"]
        data      = load_dataset("bentrevett/multi30k", split="train")
 
        src_counter = Counter()
        tgt_counter = Counter()
        for item in data:
            src_counter.update(
                t.text.lower() for t in self._spacy_de.tokenizer(item["de"])
            )
            tgt_counter.update(
                t.text.lower() for t in self._spacy_en.tokenizer(item["en"])
            )
 
        def build(counter):
            vocab = {tok: idx for idx, tok in enumerate(SPECIALS)}
            idx   = len(SPECIALS)
            for word, count in counter.items():
                if count >= freq_threshold:
                    vocab[word] = idx
                    idx += 1
            return vocab
 
        return build(src_counter), build(tgt_counter)
 
    # ── Helpers ───────────────────────────────────────────────────────
    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
 
    def _tokenize_de(self, text: str):
        return [tok.text.lower() for tok in self._spacy_de.tokenizer(text)]


    # ── AUTOGRADER HOOKS ── keep these signatures exactly ──────────────

    def encode(
        self,
        src:      torch.Tensor,
        src_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run the full encoder stack.

        Args:
            src      : Token indices, shape [batch, src_len]
            src_mask : shape [batch, 1, 1, src_len]

        Returns:
            memory : Encoder output, shape [batch, src_len, d_model]
        """
        src_emb = self.pos_encoding(self.src_embedding(src) * math.sqrt(self.d_model))
        memory = self.encoder(src_emb, src_mask)
        return memory

    def decode(
        self,
        memory:   torch.Tensor,
        src_mask: torch.Tensor,
        tgt:      torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run the full decoder stack and project to vocabulary logits.

        Args:
            memory   : Encoder output,  shape [batch, src_len, d_model]
            src_mask : shape [batch, 1, 1, src_len]
            tgt      : Token indices,   shape [batch, tgt_len]
            tgt_mask : shape [batch, 1, tgt_len, tgt_len]

        Returns:
            logits : shape [batch, tgt_len, tgt_vocab_size]
        """
        tgt_emb = self.pos_encoding(self.tgt_embedding(tgt) * math.sqrt(self.d_model))
        dec_output = self.decoder(tgt_emb, memory, src_mask, tgt_mask)
        logits = self.output_projection(dec_output)
        return logits

    def forward(
        self,
        src:      torch.Tensor,
        tgt:      torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Full encoder-decoder forward pass.

        Args:
            src      : shape [batch, src_len]
            tgt      : shape [batch, tgt_len]
            src_mask : shape [batch, 1, 1, src_len]
            tgt_mask : shape [batch, 1, tgt_len, tgt_len]

        Returns:
            logits : shape [batch, tgt_len, tgt_vocab_size]
        """
        memory = self.encode(src, src_mask)
        logits = self.decode(memory, src_mask, tgt, tgt_mask)
        return logits

    def infer(self, src_sentence: str) -> str:
        """
        Translates a German sentence to English using greedy autoregressive decoding.
        
        Args:
            src_sentence: The raw German text.
            
            
        Returns:
            The fully translated English string, detokenized and clean.
        """
        self.eval()
        device = next(self.parameters()).device

        # Tokenize source sentence using spacy
        import spacy
        from spacy.cli import download
        try:
            spacy_de = spacy.load("de_core_news_sm")
        except OSError:
            download("de_core_news_sm")
            spacy_de = spacy.load("de_core_news_sm")

        tokens = [tok.text.lower() for tok in spacy_de.tokenizer(src_sentence)]

        # Convert to indices using src_vocab (set externally after model creation)
        unk_idx = self.src_vocab.get("<unk>", 0)
        sos_idx_src = self.src_vocab.get("<sos>", self.src_vocab.get("<bos>", 2))
        eos_idx_src = self.src_vocab.get("<eos>", 3)

        src_indices = [sos_idx_src]
        for tok in tokens:
            src_indices.append(self.src_vocab.get(tok, unk_idx))
        src_indices.append(eos_idx_src)

        src_tensor = torch.tensor(src_indices, dtype=torch.long).unsqueeze(0).to(device)
        src_mask = make_src_mask(src_tensor, pad_idx=self.pad_idx)

        # Encode
        with torch.no_grad():
            memory = self.encode(src_tensor, src_mask)

        # Greedy decode
        sos_idx_tgt = self.tgt_vocab.get("<sos>", self.tgt_vocab.get("<bos>", 2))
        eos_idx_tgt = self.tgt_vocab.get("<eos>", 3)

        ys = torch.tensor([[sos_idx_tgt]], dtype=torch.long, device=device)

        idx_to_token = {v: k for k, v in self.tgt_vocab.items()}

        max_len = 128
        for _ in range(max_len):
            tgt_mask = make_tgt_mask(ys, pad_idx=self.pad_idx)
            with torch.no_grad():
                logits = self.decode(memory, src_mask, ys, tgt_mask)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            ys = torch.cat([ys, next_token], dim=1)
            if next_token.item() == eos_idx_tgt:
                break

        # Convert to words
        output_indices = ys[0].cpu().tolist()
        words = []
        special = {"<pad>", "<sos>", "<bos>", "<eos>", "<unk>"}
        for idx in output_indices[1:]:  # skip SOS
            if idx == eos_idx_tgt:
                break
            token = idx_to_token.get(idx, "<unk>")
            if token not in special:
                words.append(token)

        return " ".join(words)
