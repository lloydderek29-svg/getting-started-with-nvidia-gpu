from gpubook.validate import parse_cuda_version

HEADER = (
    "| NVIDIA-SMI 570.86.16  Driver Version: 570.86.16"
    "  CUDA Version: 12.8 |"
)


def test_parse_cuda_version():
    assert parse_cuda_version(HEADER) == (12, 8)


def test_parse_missing_version():
    assert parse_cuda_version("no gpu here") is None
