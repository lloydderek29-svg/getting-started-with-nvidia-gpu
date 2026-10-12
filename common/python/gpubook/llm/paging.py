from __future__ import annotations


def contiguous_waste(seq_lens, reserved_len: int) -> int:
    """Tokens reserved but unused when each sequence gets
    a fixed region of reserved_len tokens."""
    return reserved_len * len(seq_lens) - sum(seq_lens)


def paged_waste(seq_lens, block: int) -> int:
    """Tokens reserved but unused with block-granular
    allocation."""
    total = 0
    for n in seq_lens:
        blocks = -(-n // block)
        total += blocks * block - n
    return total


def waste_fraction(waste: int, seq_lens, per_seq: int) -> float:
    reserved = sum(seq_lens) + waste
    return waste / reserved if reserved else 0.0
