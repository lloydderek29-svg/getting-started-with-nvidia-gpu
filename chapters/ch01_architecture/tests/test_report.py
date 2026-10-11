from gpubook.report import derive_notes, parse_smi_csv

SAMPLE = (
    "0, NVIDIA GeForce RTX 4070, 8.9, 570.86.16, "
    "12282, 4, 4, 8, 200, 3105, 10501\n"
)


def test_parse_single_gpu():
    rows = parse_smi_csv(SAMPLE)
    assert len(rows) == 1
    assert rows[0]["name"] == "NVIDIA GeForce RTX 4070"
    assert rows[0]["compute_cap"] == "8.9"


def test_narrow_pcie_link_is_flagged():
    gpu = parse_smi_csv(SAMPLE)[0]
    notes = derive_notes(gpu)
    assert any("x8" in n for n in notes)


def test_runtime_newer_than_driver_is_flagged():
    gpu = {
        "pcie.link.width.current": "16",
        "runtime": {"runtime_version": 12080,
                    "driver_version": 12040},
    }
    assert derive_notes(gpu)
