import argparse
import time
import urllib.request

from gpubook.serving.metrics import (
    gpu_summary, model_rates, parse_prometheus, per_model)


def scrape(url: str):
    with urllib.request.urlopen(url, timeout=3) as r:
        return parse_prometheus(r.read().decode())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url",
                    default="http://localhost:8002/metrics")
    ap.add_argument("--interval", type=float, default=2.0)
    a = ap.parse_args()

    before, t0 = scrape(a.url), time.time()
    print(f"{'model':16s} {'req/s':>7s} {'fail/s':>7s} "
          f"{'batch':>6s} {'lat ms':>7s} {'queue':>7s} "
          f"{'compute':>8s} {'pend':>5s}")
    while True:
        time.sleep(a.interval)
        after, t1 = scrape(a.url), time.time()
        rates = model_rates(before, after, t1 - t0)
        pending = per_model(
            after, "nv_inference_pending_request_count")
        for name, r in sorted(rates.items()):
            if r["rps"] + r["fail_rps"] == 0:
                continue
            print(f"{name:16s} {r['rps']:7.1f} "
                  f"{r['fail_rps']:7.2f} "
                  f"{r['avg_batch']:6.1f} "
                  f"{r['latency_ms']:7.2f} "
                  f"{r['queue_ms']:7.2f} "
                  f"{r['compute_ms']:8.2f} "
                  f"{pending.get(name, 0):5.0f}")
        g = gpu_summary(after)
        print(f"{'gpu':16s} util {g['util_pct']:5.1f} %  "
              f"mem {g['mem_gib']:5.2f} GiB  "
              f"power {g['power_w']:6.1f} W")
        before, t0 = after, t1


if __name__ == "__main__":
    main()
