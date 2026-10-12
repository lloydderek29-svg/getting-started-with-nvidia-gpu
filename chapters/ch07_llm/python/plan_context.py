from gpubook.llm.memory import ModelShape, plan_table

SHAPE = ModelShape(
    "8B-class", 8_030_000_000, 32, 8, 128, 131072)

for batch in (1, 8):
    print(f"\nconcurrent sequences: {batch}")
    print(f"{'VRAM':>5s} {'bits':>5s} "
          f"{'weights':>9s} {'context':>9s}")
    for r in plan_table(SHAPE, [12, 16, 24, 48, 80],
                        [16, 8, 4], batch):
        print(f"{r['vram_gib']:5d} {r['weight_bits']:5d} "
              f"{r['weights_gib']:7.1f}Gi {r['context']:9d}")
