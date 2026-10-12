import torch.nn as nn

from gpubook.genai.pipelines import component_sizes


class FakePipe:
    components = {
        "unet": nn.Linear(10, 10),
        "scheduler": object(),
        "vae": nn.Linear(4, 4, bias=False),
    }


def test_component_sizes_count_modules_only():
    sizes = component_sizes(FakePipe())
    assert set(sizes) == {"unet", "vae"}
    assert sizes["unet"] == (110, 110 * 4)
    assert sizes["vae"] == (16, 64)
