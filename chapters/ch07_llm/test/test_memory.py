from gpubook.llm.memory import (
    GIB, ModelShape, kv_bytes_per_token, max_tokens)

LLAMA_8B_CFG = {
    "hidden_size": 4096,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "num_hidden_layers": 32,
    "max_position_embeddings": 131072,
}


def test_llama_8b_cache_per_token():
    s = ModelShape.from_config(LLAMA_8B_CFG, 8_030_000_000)
    assert s.head_dim == 128
    assert kv_bytes_per_token(s, 16) == 131072
    assert kv_bytes_per_token(s, 8) == 65536


def test_missing_kv_heads_means_full_attention():
    cfg = dict(LLAMA_8B_CFG)
    del cfg["num_key_value_heads"]
    s = ModelShape.from_config(cfg, 1)
    assert s.kv_heads == 32


def test_max_tokens_matches_hand_calculation():
    s = ModelShape("t", 10**9, 2, 1, 64, 0)
    assert kv_bytes_per_token(s) == 512
    got = max_tokens(s, 2, 8, overhead_gib=0)
    assert got == int((2 * GIB * 0.9 - 1e9) // 512)


def test_model_that_does_not_fit_has_zero_context():
    s = ModelShape("t", 10**9, 2, 1, 64, 0)
    assert max_tokens(s, 2, 16, overhead_gib=0) == 0
