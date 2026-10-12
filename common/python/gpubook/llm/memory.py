from __future__ import annotations

from dataclasses import dataclass

GIB = 2**30


@dataclass(frozen=True)
class ModelShape:
    name: str
    params: int
    layers: int
    kv_heads: int
    head_dim: int
    max_positions: int

    @classmethod
    def from_config(cls, cfg: dict, params: int,
                    name: str = "model") -> "ModelShape":
        cfg = cfg.get("text_config", cfg)
        heads = cfg["num_attention_heads"]
        kv = cfg.get("num_key_value_heads") or heads
        head_dim = cfg.get("head_dim") or cfg["hidden_size"] // heads
        return cls(
            name=name,
            params=params,
            layers=cfg["num_hidden_layers"],
            kv_heads=kv,
            head_dim=head_dim,
            max_positions=cfg.get("max_position_embeddings", 0),
        )


def weights_bytes(params: int, bits: float) -> float:
    return params * bits / 8


def kv_bytes_per_token(s: ModelShape, kv_bits: int = 16) -> int:
    return 2 * s.layers * s.kv_heads * s.head_dim * kv_bits // 8


def kv_cache_bytes(s: ModelShape, tokens: int, batch: int = 1,
                   kv_bits: int = 16) -> int:
    return kv_bytes_per_token(s, kv_bits) * tokens * batch


def max_tokens(s: ModelShape, vram_gib: float, weight_bits: float,
               batch: int = 1, kv_bits: int = 16,
               overhead_gib: float = 1.5,
               headroom: float = 0.9) -> int:
    """Largest context the cache allows, per sequence."""
    budget = vram_gib * GIB * headroom
    budget -= weights_bytes(s.params, weight_bits)
    budget -= overhead_gib * GIB
    if budget <= 0:
        return 0
    return int(budget // (kv_bytes_per_token(s, kv_bits) * batch))


def usable_context(s: ModelShape, **kwargs) -> int:
    limit = max_tokens(s, **kwargs)
    if s.max_positions:
        limit = min(limit, s.max_positions)
    return limit


def plan_table(s: ModelShape, vram_list, bits_list, batch=1):
    rows = []
    for vram in vram_list:
        for bits in bits_list:
            rows.append({
                "vram_gib": vram,
                "weight_bits": bits,
                "weights_gib": weights_bytes(s.params, bits) / GIB,
                "context": usable_context(
                    s, vram_gib=vram, weight_bits=bits,
                    batch=batch),
            })
    return rows
