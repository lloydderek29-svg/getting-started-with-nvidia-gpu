from gpubook.monitor import decode_throttle


def test_decode_multiple_reasons():
    mask = 0x4 | 0x20
    assert decode_throttle(mask) == [
        "sw_power_cap", "sw_thermal"]


def test_zero_mask_has_no_reasons():
    assert decode_throttle(0) == []
