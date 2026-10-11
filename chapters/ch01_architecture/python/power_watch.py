import time

import pynvml


def main(seconds: int = 10) -> None:
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    print("time_s  power_W  sm_MHz  temp_C")
    t0 = time.time()
    while time.time() - t0 < seconds:
        mw = pynvml.nvmlDeviceGetPowerUsage(handle)
        sm = pynvml.nvmlDeviceGetClockInfo(
            handle, pynvml.NVML_CLOCK_SM
        )
        temp = pynvml.nvmlDeviceGetTemperature(
            handle, pynvml.NVML_TEMPERATURE_GPU
        )
        elapsed = time.time() - t0
        print(f"{elapsed:6.1f}  {mw / 1000:7.1f}  "
              f"{sm:6d}  {temp:6d}")
        time.sleep(1)
    pynvml.nvmlShutdown()


if __name__ == "__main__":
    main()
