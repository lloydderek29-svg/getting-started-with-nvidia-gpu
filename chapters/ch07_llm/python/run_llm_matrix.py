import json
import subprocess
import sys
from pathlib import Path

BENCH = Path(__file__).with_name("llm_bench.py")
OUT = Path("results/ch07/matrix")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for quant in ("bf16", "int8", "nf4"):
        path = OUT / f"{quant}.json"
        if not path.exists():
            subprocess.run(
                [sys.executable, str(BENCH), "--quant", quant,
                 "--batches", "1,8", "--out", str(path)],
                check=True)
        data = json.loads(path.read_text())
        for r in data["results"]:
            if not r.get("oom"):
                rows.append((quant, r))

    print(f"{'quant':6s} {'batch':>5s} {'TTFT ms':>8s} "
          f"{'tok/s all':>10s} {'GiB':>6s}")
    for quant, r in rows:
        print(f"{quant:6s} {r['batch']:5d} {r['ttft_ms']:8.0f} "
              f"{r['decode_tps_total']:10.0f} "
              f"{r['peak_gib']:6.2f}")


if __name__ == "__main__":
    main()
