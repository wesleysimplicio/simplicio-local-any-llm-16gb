#!/usr/bin/env python3
"""Generates the toy BPE tokenizer fixture used by bpe_tokenizer_contract_test.

Trains a real byte-pair-encoding model (Sennrich et al. algorithm: repeatedly
merge the most frequent adjacent symbol pair) on a small toy corpus, emits a
HuggingFace `tokenizers`-schema tokenizer.json, then independently re-derives
expected token ids for a fixed set of sentences using the SAME merge rules as
an out-of-repo oracle for the C++ implementation under test.

Run manually to regenerate the checked-in fixtures:
    python3 tests/fixtures/tokenizer/generate_toy_bpe.py
"""
import json
import os
from collections import Counter

CORPUS = {
    "low": 5,
    "lower": 2,
    "lowest": 2,
    "newer": 6,
    "wider": 3,
    "new": 2,
}

NUM_MERGES = 12
UNK_TOKEN = "<unk>"


def word_to_symbols(word):
    return list(word)


def get_pair_counts(word_symbols):
    counts = Counter()
    for word, symbols in word_symbols.items():
        freq = CORPUS[word]
        for a, b in zip(symbols, symbols[1:]):
            counts[(a, b)] += freq
    return counts


def merge_pair(word_symbols, pair):
    a, b = pair
    merged = a + b
    for word, symbols in word_symbols.items():
        new_symbols = []
        i = 0
        while i < len(symbols):
            if i + 1 < len(symbols) and symbols[i] == a and symbols[i + 1] == b:
                new_symbols.append(merged)
                i += 2
            else:
                new_symbols.append(symbols[i])
                i += 1
        word_symbols[word] = new_symbols


def train_bpe():
    word_symbols = {word: word_to_symbols(word) for word in CORPUS}
    merges = []
    for _ in range(NUM_MERGES):
        counts = get_pair_counts(word_symbols)
        if not counts:
            break
        best_pair = max(counts.items(), key=lambda kv: (kv[1], kv[0]))[0]
        merges.append(best_pair)
        merge_pair(word_symbols, best_pair)
    return merges, word_symbols


def build_vocab(merges, word_symbols):
    vocab = {}
    next_id = 0

    def add(token):
        nonlocal next_id
        if token not in vocab:
            vocab[token] = next_id
            next_id += 1

    add(UNK_TOKEN)
    for word in CORPUS:
        for ch in word:
            add(ch)
    for a, b in merges:
        add(a + b)
    return vocab


def bpe_encode_word(word, merge_rank):
    symbols = list(word)
    while len(symbols) > 1:
        best_rank = None
        best_index = None
        for i in range(len(symbols) - 1):
            pair = (symbols[i], symbols[i + 1])
            rank = merge_rank.get(pair)
            if rank is not None and (best_rank is None or rank < best_rank):
                best_rank = rank
                best_index = i
        if best_index is None:
            break
        symbols[best_index] = symbols[best_index] + symbols[best_index + 1]
        del symbols[best_index + 1]
    return symbols


def encode_sentence(sentence, merge_rank, vocab):
    ids = []
    for word in sentence.split():
        for token in bpe_encode_word(word, merge_rank):
            ids.append(vocab.get(token, vocab[UNK_TOKEN]))
    return ids


def main():
    merges, _ = train_bpe()
    vocab = build_vocab(merges, {})
    merge_rank = {pair: rank for rank, pair in enumerate(merges)}

    tokenizer_json = {
        "model": {
            "type": "BPE",
            "vocab": vocab,
            "merges": [f"{a} {b}" for a, b in merges],
            "unk_token": UNK_TOKEN,
        }
    }

    sentences = [
        "low lower lowest",
        "newer wider new",
        "low newer unseenword",
    ]
    reference = {
        sentence: encode_sentence(sentence, merge_rank, vocab)
        for sentence in sentences
    }

    out_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(out_dir, "toy_bpe_tokenizer.json"), "w") as f:
        json.dump(tokenizer_json, f, indent=2, sort_keys=True)
        f.write("\n")
    with open(os.path.join(out_dir, "reference_output.json"), "w") as f:
        json.dump(reference, f, indent=2, sort_keys=True)
        f.write("\n")

    print(f"wrote {len(vocab)} vocab entries, {len(merges)} merges, "
          f"{len(reference)} reference sentences")


if __name__ == "__main__":
    main()
