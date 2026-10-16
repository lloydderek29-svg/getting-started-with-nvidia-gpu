# Chapter 7: LLM Inference and Memory Optimization

This directory loads local language models, predicts and measures their memory use, applies quantization, speeds up generation, and benchmarks time to first token, inter-token latency, throughput, and memory. Shared code is in `common/python/gpubook/llm/`. Run every command from the repository root.

## What Is Here

| Path | Purpose |
|---|---|
| `python/preflight.py` | Reads a model's files and judges GPU fit before loading |
| `python/chat.py` | Loads a model and answers one prompt |
| `python/plan_context.py` | Prints maximum context by VRAM and precision |
| `python/kv_measure.py` | Compares predicted and measured KV cache memory |
| `python/quant_compare.py` | bf16, 8-bit, and 4-bit memory, perplexity, and speed |
| `python/batch_sweep.py` | Throughput against batch size |
| `python/attn_compare.py` | Eager, SDPA, and FlashAttention-2 prefill |
| `python/prefix_cache.py` | Time to first token with and without a prefix cache |
| `python/assisted.py` | Speculative decoding with a draft model |
| `python/llm_bench.py` | Full latency and throughput benchmark |
| `python/run_llm_matrix.py` | Runs the benchmark per precision in clean processes |

## Prerequisites

```bash
pip install -r requirements/torch.txt
pip install -r requirements/llm.txt
pip install -e .
```

The default model is `Qwen/Qwen2.5-1.5B-Instruct`, and every script accepts `--model`. Some models require accepting a license on the Hugging Face website and running `huggingface-cli login`. FlashAttention-2 needs the separate `flash-attn` package and an Ampere or newer GPU, and the comparison script reports it as unavailable otherwise. The 4-bit and 8-bit modes need `bitsandbytes`.

## Run

Check fit before loading weights onto the GPU.

```bash
python chapters/ch07_llm/python/preflight.py Qwen/Qwen2.5-1.5B-Instruct
```

The report lists parameters, stored size, context limit, GPU compatibility, and a fits or does not fit verdict at 16, 8, and 4 bits for weights alone.

```bash
python chapters/ch07_llm/python/chat.py "Explain what a KV cache is."
python chapters/ch07_llm/python/plan_context.py
python chapters/ch07_llm/python/kv_measure.py \
  --lengths 512,2048,8192 --find-limit
```

In `kv_measure.py`, the ratio of measured to predicted cache should sit very close to 1.00. The empirical limit is below the cache-only prediction because prefill creates large temporary tensors.

```bash
python chapters/ch07_llm/python/quant_compare.py
python chapters/ch07_llm/python/batch_sweep.py
python chapters/ch07_llm/python/attn_compare.py --tokens 4096
python chapters/ch07_llm/python/prefix_cache.py
```

Expect memory to fall by about half at 8-bit and by more than three quarters at 4-bit, with perplexity rising slightly at 4-bit. Total throughput should rise almost linearly with batch size before flattening. The prefix cache should cut time to first token for long shared prefixes.

Speculative decoding needs two models that share a tokenizer. On a smaller card, use the 1.5B model as target and the 0.5B model as draft.

```bash
python chapters/ch07_llm/python/assisted.py \
  --target Qwen/Qwen2.5-1.5B-Instruct \
  --draft Qwen/Qwen2.5-0.5B-Instruct
```

## Benchmark

```bash
python chapters/ch07_llm/python/llm_bench.py \
  --quant nf4 --batches 1,4,16 \
  --bandwidth-gbs 504 --out results/ch07/nf4.json

python chapters/ch07_llm/python/run_llm_matrix.py
```

Pass the memory bandwidth you measured in Chapter 1. The output includes TTFT, mean and p95 inter-token latency, per-request and total tokens per second, peak memory, and energy per token. The ceiling line compares batch-one decode speed with the bandwidth limit, and a result far below it suggests launch overhead.

## Tests

```bash
pytest chapters/ch07_llm -v
```

The tests cover the safetensors header parser, the memory arithmetic, perplexity windows, the paging simulation, and the timing summary. None of them needs a GPU.

## Troubleshooting

| Problem | Fix |
|---|---|
| Download is unauthorized | Accept the model license and log in |
| Out of memory on load | Use `--quant` or a smaller model |
| bitsandbytes import errors | Match its build to your CUDA version |
| Garbage output in batches | Padding must be on the left, which the loader sets |
| Loaded dtype error | The loader checks it, so check your transformers version |