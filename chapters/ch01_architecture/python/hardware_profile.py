import os

import pynvml


def is_jetson() -> bool:
    return os.path.exists("/etc/nv_tegra_release")


def _safe(fn, *args):
    """Call an NVML function, return None if unsupported."""
    try:
        return fn(*args)
    except pynvml.NVMLError:
        return None


def _text(value):
    return value.decode() if isinstance(value, bytes) else value


def describe(index: int) -> dict:
    handle = pynvml.nvmlDeviceGetHandleByIndex(index)
    mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
    ecc = _safe(pynvml.nvmlDeviceGetEccMode, handle)
    mig = _safe(pynvml.nvmlDeviceGetMigMode, handle)
    return {
        "index": index,
        "name": _text(pynvml.nvmlDeviceGetName(handle)),
        "vram_gib": round(mem.total / 2**30, 1),
        "ecc_enabled": bool(ecc[0]) if ecc else None,
        "mig_enabled": bool(mig[0]) if mig else None,
    }


def main() -> None:
    if is_jetson():
        print("Jetson platform detected (shared memory SoC)")
    pynvml.nvmlInit()
    try:
        for i in range(pynvml.nvmlDeviceGetCount()):
            print(describe(i))
    finally:
        pynvml.nvmlShutdown()


if __name__ == "__main__":
    main()
