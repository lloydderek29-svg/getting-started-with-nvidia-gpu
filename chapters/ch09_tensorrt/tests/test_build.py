from gpubook.trt.build import engine_name


def test_engine_name_encodes_the_target():
    name = engine_name("pets", "fp16", "8.9", "10.3.0", 1, 8, 64)
    assert name == "pets_fp16_sm89_trt10.3.0_b1-8-64.engine"
