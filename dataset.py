"""
dataset.py — Multi30k Dataset Loading and Spacy Tokenization
DA6401 Assignment 3
"""

import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
from datasets import load_dataset


class Multi30kDataset:
    """
    Loads the Multi30k dataset and prepares tokenizers, vocabularies,
    and numericalized (integer-indexed) sentence pairs.

    Special tokens and their default indices:
        <unk> = 0,  <pad> = 1,  <sos> = 2,  <eos> = 3

    Usage:
        ds = Multi30kDataset(split='train')
        ds.build_vocab()
        ds.process_data()
        # Access: ds.src_data, ds.tgt_data (lists of LongTensors)
        #         ds.src_vocab, ds.tgt_vocab  (token→idx dicts)
        #         ds.src_itos,  ds.tgt_itos   (idx→token dicts)
    """

    # Special token indices (fixed)
    UNK_IDX = 0
    PAD_IDX = 1
    SOS_IDX = 2
    EOS_IDX = 3
    SPECIALS = ["<unk>", "<pad>", "<sos>", "<eos>"]

    def __init__(self, split='train', freq_threshold=2):
        """
        Loads the Multi30k dataset and prepares tokenizers.
        """
        self.split = split
        self.freq_threshold = freq_threshold

        # Load dataset from Hugging Face
        self.dataset = load_dataset("bentrevett/multi30k", split=split)

        # Load spacy tokenizers
        import spacy
        try:
            self.spacy_de = spacy.load("de_core_news_sm")
        except OSError:
            import os
            os.system("python -m spacy download de_core_news_sm")
            self.spacy_de = spacy.load("de_core_news_sm")

        try:
            self.spacy_en = spacy.load("en_core_web_sm")
        except OSError:
            import os
            os.system("python -m spacy download en_core_web_sm")
            self.spacy_en = spacy.load("en_core_web_sm")

        # Will be populated by build_vocab / process_data
        self.src_vocab = None   # token → idx
        self.tgt_vocab = None
        self.src_itos = None    # idx → token
        self.tgt_itos = None
        self.src_data = None    # list of LongTensors
        self.tgt_data = None

    def tokenize_de(self, text):
        """Tokenize German text."""
        return [tok.text.lower() for tok in self.spacy_de.tokenizer(text)]

    def tokenize_en(self, text):
        """Tokenize English text."""
        return [tok.text.lower() for tok in self.spacy_en.tokenizer(text)]

    def build_vocab(self, train_dataset=None):
        """
        Builds the vocabulary mapping for src (de) and tgt (en), including:
        <unk>, <pad>, <sos>, <eos>

        Args:
            train_dataset: If provided, use this dataset to build vocab
                           (useful for val/test splits that should use
                            the training vocab). If None, uses self.dataset.
        """
        data = train_dataset if train_dataset is not None else self.dataset

        # Count token frequencies
        src_counter = Counter()
        tgt_counter = Counter()
        for item in data:
            src_counter.update(self.tokenize_de(item["de"]))
            tgt_counter.update(self.tokenize_en(item["en"]))

        # Build src vocab
        self.src_vocab = {tok: idx for idx, tok in enumerate(self.SPECIALS)}
        idx = len(self.SPECIALS)
        for word, count in src_counter.items():
            if count >= self.freq_threshold:
                self.src_vocab[word] = idx
                idx += 1
        self.src_itos = {v: k for k, v in self.src_vocab.items()}

        # Build tgt vocab
        self.tgt_vocab = {tok: idx for idx, tok in enumerate(self.SPECIALS)}
        idx = len(self.SPECIALS)
        for word, count in tgt_counter.items():
            if count >= self.freq_threshold:
                self.tgt_vocab[word] = idx
                idx += 1
        self.tgt_itos = {v: k for k, v in self.tgt_vocab.items()}

    def set_vocab(self, src_vocab, tgt_vocab, src_itos, tgt_itos):
        """Set vocabularies from an external source (e.g., training set)."""
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.src_itos = src_itos
        self.tgt_itos = tgt_itos

    def process_data(self):
        """
        Convert English and German sentences into integer token lists using
        spacy and the defined vocabulary.

        Populates self.src_data and self.tgt_data as lists of LongTensors,
        each wrapped with <sos> ... <eos>.
        """
        assert self.src_vocab is not None, "Call build_vocab() first!"

        self.src_data = []
        self.tgt_data = []

        for item in self.dataset:
            # Tokenize
            src_tokens = self.tokenize_de(item["de"])
            tgt_tokens = self.tokenize_en(item["en"])

            # Numericalize with SOS/EOS
            src_indices = (
                [self.SOS_IDX]
                + [self.src_vocab.get(tok, self.UNK_IDX) for tok in src_tokens]
                + [self.EOS_IDX]
            )
            tgt_indices = (
                [self.SOS_IDX]
                + [self.tgt_vocab.get(tok, self.UNK_IDX) for tok in tgt_tokens]
                + [self.EOS_IDX]
            )

            self.src_data.append(torch.tensor(src_indices, dtype=torch.long))
            self.tgt_data.append(torch.tensor(tgt_indices, dtype=torch.long))

    def __len__(self):
        if self.src_data is not None:
            return len(self.src_data)
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.src_data[idx], self.tgt_data[idx]


# Collate & DataLoader helpers

def collate_fn(batch, pad_idx=1):
    """Pad source and target sequences to the longest in the batch."""
    src_batch, tgt_batch = zip(*batch)
    src_batch = pad_sequence(src_batch, batch_first=True, padding_value=pad_idx)
    tgt_batch = pad_sequence(tgt_batch, batch_first=True, padding_value=pad_idx)
    return src_batch, tgt_batch


def prepare_data(batch_size=128, freq_threshold=2):
    """
    Full data preparation pipeline.

    Returns:
        train_loader, val_loader, test_loader,
        train_ds (Multi30kDataset with vocab attached)
    """
    # Create datasets for each split
    train_ds = Multi30kDataset(split="train", freq_threshold=freq_threshold)
    val_ds = Multi30kDataset(split="validation", freq_threshold=freq_threshold)
    test_ds = Multi30kDataset(split="test", freq_threshold=freq_threshold)

    # Build vocab from training data
    train_ds.build_vocab()

    # Share vocab with val/test
    val_ds.set_vocab(train_ds.src_vocab, train_ds.tgt_vocab, train_ds.src_itos, train_ds.tgt_itos)
    test_ds.set_vocab(train_ds.src_vocab, train_ds.tgt_vocab, train_ds.src_itos, train_ds.tgt_itos)

    # Numericalize
    train_ds.process_data()
    val_ds.process_data()
    test_ds.process_data()

    print(f"Source (German) vocabulary size: {len(train_ds.src_vocab)}")
    print(f"Target (English) vocabulary size: {len(train_ds.tgt_vocab)}")
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    pad_idx = Multi30kDataset.PAD_IDX

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=lambda b: collate_fn(b, pad_idx), num_workers=0, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=lambda b: collate_fn(b, pad_idx), num_workers=0, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        collate_fn=lambda b: collate_fn(b, pad_idx), num_workers=0, pin_memory=True,
    )

    return train_loader, val_loader, test_loader, train_ds
