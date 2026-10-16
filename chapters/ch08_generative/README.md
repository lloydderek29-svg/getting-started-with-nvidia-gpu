# Chapter 8: Generative AI and Efficient Model Fine-Tuning

This directory runs diffusion pipelines, measures their memory behavior, trains a LoRA adapter, compares speedups against quality, and builds a reproducible generation workflow. Shared code is in `common/python/gpubook/genai/`. Run every command from the repository root.

## What Is Here

| Path | Purpose |
|---|---|
| `python/generate_basic.py` | One image and a component size listing |
| `python/sched_compare.py` | Scheduler and step comparison contact sheet |
| `python/vram_sweep.py` | Offload, tiling, and resolution against peak memory |
| `python/train_lora.py` | LoRA training on a captioned folder |
| `python/lora_infer.py` | Generates at several adapter strengths |
| `python/gen_bench.py` | Speed and quality comparison with a Pareto front |
| `python/generate.py` | Config-driven batch generation with metadata |
| `python/verify_repro.py` | Regenerates an image from its embedded settings |
| `configs/gen_default.yaml` | Example run configuration |
| `configs/prompts.txt` | Optional scratch list of prompts, since the generator reads prompts from the YAML file |

## Prerequisites

```bash
pip install -r requirements/torch.txt
pip install -r requirements/genai.txt
pip install -e .
```

Two models are used: `stabilityai/stable-diffusion-xl-base-1.0` for generation and `stable-diffusion-v1-5/stable-diffusion-v1-5` for LoRA training. Confirm each model's current location and license. Add `chapters/ch08_generative/python` to the pytest path so tests can import the scripts.

## Run

```bash
python chapters/ch08_generative/python/generate_basic.py \
  "a lighthouse at dawn, oil painting"

python chapters/ch08_generative/python/sched_compare.py \
  "a lighthouse at dawn, oil painting"

python chapters/ch08_generative/python/vram_sweep.py \
  --sizes 768,1024,1536
```

The first call is slower than later ones because kernels initialize lazily. The sweep prints seconds and peak memory per setting, with out-of-memory cases shown as data. A black result is flagged, which usually means FP16 overflow in the SDXL VAE, and the `--vae madebyollin/sdxl-vae-fp16-fix` option avoids it.

## LoRA Training

Create a folder of 20 to 50 images, each with a text file of the same name containing its caption. For a style, put a rare trigger word in every caption.

```text
data/inkwash/
  001.jpg  001.txt
  002.jpg  002.txt
```

```bash
python chapters/ch08_generative/python/train_lora.py \
  --data data/inkwash --steps 600

python chapters/ch08_generative/python/lora_infer.py \
  "inkwash style, a heron at the water" \
  --weights results/ch08/lora
```

Training prints the trainable parameter count, a running loss, and the peak memory, which typically fits an 8 GB card with the default batch and accumulation. Checkpoints are saved every 200 steps under `results/ch08/lora/`. The inference script writes a grid at strengths 0.0, 0.5, and 1.0 with a fixed seed. If strength 1.0 copies a training image regardless of the prompt, the adapter is overfit, so use an earlier checkpoint.

## Speed and Quality

```bash
python chapters/ch08_generative/python/gen_bench.py
```

The table lists seconds per image, memory, CLIP text score, CLIP similarity to the 50-step reference, PSNR, and whether a configuration is on the Pareto front. The metrics are proxies. Open `results/ch08/bench_sheet.png` and judge by eye as well.

## Reproducible Workflow

```bash
python chapters/ch08_generative/python/generate.py \
  --config chapters/ch08_generative/configs/gen_default.yaml

python chapters/ch08_generative/python/verify_repro.py \
  outputs/lighthouse_set/images/000_*.png
```

The run directory contains `config.yaml`, `manifest.jsonl`, `summary.json`, and an `images/` folder. Each PNG file name carries a hash of its settings, and the settings are embedded in the file. Interrupted runs resume by skipping images that exist. On the same machine and library versions, the verification script should report a maximum pixel difference of zero. Across machines or upgrades, small differences are normal, and the recorded environment shows what changed.

## Tests

```bash
pytest chapters/ch08_generative -v
```

## Troubleshooting

| Problem | Fix |
|---|---|
| Out of memory at the final step | Add VAE tiling or model offload |
| Very slow with sequential offload | Expected, so use it as a last resort |
| LCM images look different | It is a different look, not only lower quality |
| Different pixels on another machine | Different kernels, so pin versions for bitwise results |