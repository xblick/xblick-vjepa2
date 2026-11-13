"""model.py against transformers' own pipeline (VJEPA2VideoProcessor, then VJEPA2ForVideoClassification): the same
inputs from the same frames, the same tokens' shape and the same class probabilities. Needs a GPU and weights/ssv2
(tools/setup.sh)."""

import numpy as np
import pytest
import torch

from xblick_vjepa2.model import WEIGHTS, VJepa2

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or not (WEIGHTS / "ssv2" / "model.safetensors").is_file(),
                                reason="needs a GPU and weights/ssv2 (tools/setup.sh)")


def frames(n: int, h: int, w: int) -> np.ndarray:
    """A clip of a smooth pattern moving to the right, with some texture: (n, h, w, 3) uint8."""
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    out = []
    for t in range(n):
        r = 127 + 120 * np.sin((x - 12 * t) / 37.0)
        g = 127 + 120 * np.cos(y / 23.0 + t / 5.0)
        b = 127 + 120 * np.sin((x + y) / 51.0)
        out.append(np.stack([r, g, b], -1))
    return np.clip(np.stack(out), 0, 255).astype(np.uint8)


@pytest.fixture(scope="module")
def model():
    return VJepa2("ssv2", "cuda")


@pytest.fixture(scope="module")
def reference():
    from transformers import VJEPA2ForVideoClassification, VJEPA2VideoProcessor

    full = VJEPA2ForVideoClassification.from_pretrained(WEIGHTS / "ssv2", dtype=torch.bfloat16,
                                                        attn_implementation="sdpa").cuda().eval()
    return VJEPA2VideoProcessor(), full


@pytest.mark.parametrize("size", [(480, 640), (720, 1280), (640, 480), (300, 300)])
def test_inputs_are_the_processors(model, reference, size):
    processor, _ = reference
    video = frames(2, *size)
    theirs = processor(list(video), return_tensors="pt")["pixel_values_videos"][0].cuda()  # (2, 3, 256, 256)
    ours = torch.stack([model.prepare(torch.as_tensor(f, device="cuda")) for f in video]).float()
    assert ours.shape == theirs.shape
    diff = (ours - theirs.float()).abs()
    # bfloat16 (8 bits of mantissa) and the resize's rounding: within a grey level or two
    assert diff.mean() < 0.01 and diff.max() < 0.1, (diff.mean().item(), diff.max().item())


def test_tokens_and_probabilities_are_the_models(model, reference):
    processor, full = reference
    video = frames(16, 480, 640)
    inputs = [model.prepare(torch.as_tensor(f, device="cuda")) for f in video]
    tokens = model.encode(inputs)
    assert tokens.shape == (model.tokens_of(16), 1024) == (2048, 1024) and tokens.dtype == torch.bfloat16
    ours = model.classify(tokens)
    with torch.inference_mode():
        logits = full(pixel_values_videos=torch.stack(inputs)[None]).logits[0].float()
    assert torch.allclose(ours, logits.softmax(-1), atol=1e-3)
    assert len(model.labels) == 174 and abs(ours.sum().item() - 1) < 1e-4
