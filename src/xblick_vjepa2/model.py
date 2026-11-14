"""V-JEPA 2 (Meta, 2025; transformers' VJEPA2ForVideoClassification): its frozen video encoder and an attentive head
that reads the encoder's tokens.

* ``prepare``: one camera frame as the model's input, on the GPU, as transformers' VJEPA2VideoProcessor makes it
  (the short side scaled to 256 * 256 / 224 = 292, the centre 256 x 256 cut out, ImageNet's mean and deviation);
  each frame once, when it comes, so a clip is only stacked
* ``encode``: a clip (``frames`` inputs, oldest first) -> the encoder's last tokens, (N, 1024) for ViT-L: N =
  frames / 2 x 16 x 16, one token per two frames (a tubelet) and 16 x 16 pixels, in the order time, row, column; what
  an attentive head reads (``VJEPA2AttentivePooler`` takes them as they are)
* ``classify``: the head's class probabilities (softmax) for those tokens

The checkpoint is a VJEPA2ForVideoClassification (encoder, pooler, classifier): Meta's SSv2 one by default, whose
encoder is exactly the pretrained one (Meta trains only the head; their checkpoints' encoder weights are the same bits
as vjepa2-vitl-fpc64-256's).
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from .weights import CHECKPOINT_FILES, CHECKPOINTS  # the checkpoints' one list (weights.py)

IMAGENET_MEAN = (0.485, 0.456, 0.406)  # transformers' IMAGENET_DEFAULT_MEAN and _STD, as VJEPA2VideoProcessor uses
IMAGENET_STD = (0.229, 0.224, 0.225)


def checkpoint_dir(name: str, root: Path) -> Path:
    """<root>/<name> of a known checkpoint, else ``name`` as a directory of a VJEPA2ForVideoClassification."""
    path = Path(root) / name if name in CHECKPOINTS else Path(name)
    missing = [f for f in CHECKPOINT_FILES if not (path / f).is_file()]
    if missing:
        hint = " (xblick-vjepa2 downloads it at its start, or with --download-only)" if name in CHECKPOINTS else ""
        raise FileNotFoundError(f"checkpoint {name}: {', '.join(missing)} not in {path}{hint}")
    return path


class VJepa2:
    """The encoder and head of a checkpoint, in bfloat16 on ``device``. ``labels``: False leaves the head out."""

    def __init__(self, checkpoint: str, root: Path, device="cuda", labels: bool = True):
        from transformers import VJEPA2ForVideoClassification  # noqa: PLC0415 - slow; after the arguments

        model = VJEPA2ForVideoClassification.from_pretrained(checkpoint_dir(checkpoint, root), dtype=torch.bfloat16,
                                                             attn_implementation="sdpa")
        model.vjepa2.predictor = None  # the pretraining's predictor: not used for features
        self.device = torch.device(device)
        self.encoder = model.vjepa2.encoder.to(self.device).eval()
        self.pooler = model.pooler.to(self.device).eval() if labels else None
        self.classifier = model.classifier.to(self.device).eval() if labels else None
        config = model.config
        self.labels = [config.id2label[i] for i in range(config.num_labels)] if labels else []
        self.size = config.crop_size  # 256
        self.resize = int(self.size * 256 / 224)  # the processor's short side: 292
        self.patch, self.tubelet, self.width = config.patch_size, config.tubelet_size, config.hidden_size
        self.frames_trained = config.frames_per_clip  # the head's clip length (16 for SSv2)
        self._mean = torch.tensor(IMAGENET_MEAN, device=self.device).view(3, 1, 1) * 255
        self._std = torch.tensor(IMAGENET_STD, device=self.device).view(3, 1, 1) * 255

    def tokens_of(self, frames: int) -> int:
        """How many tokens a clip of ``frames`` gives."""
        return frames // self.tubelet * (self.size // self.patch) ** 2

    @torch.inference_mode()
    def prepare(self, rgb: torch.Tensor) -> torch.Tensor:
        """An (H, W, 3) uint8 RGB frame -> the model's (3, 256, 256) bfloat16 input of it."""
        h, w = rgb.shape[:2]
        # transformers' get_resize_output_image_size (shortest_edge): the long side truncated
        nh, nw = (self.resize, int(self.resize * w / h)) if h <= w else (int(self.resize * h / w), self.resize)
        x = rgb.permute(2, 0, 1)[None].float()
        # torchvision's resize of uint8 frames (the processor's): antialiased bilinear, rounded to integers
        x = F.interpolate(x, size=(nh, nw), mode="bilinear", antialias=True, align_corners=False)[0]
        x = x.round_().clamp_(0, 255)
        top, left = (nh - self.size) // 2, (nw - self.size) // 2  # the processor's centre: odd margins rounded down
        x = x[:, top:top + self.size, left:left + self.size]
        return ((x - self._mean) / self._std).to(torch.bfloat16)

    @torch.inference_mode()
    def encode(self, inputs: list[torch.Tensor]) -> torch.Tensor:
        """A clip's inputs (``prepare``'s, oldest first) -> the encoder's tokens, (N, width) bfloat16."""
        video = torch.stack(inputs)[None]  # (1, frames, 3, 256, 256), as pixel_values_videos
        return self.encoder(pixel_values_videos=video).last_hidden_state[0]

    @torch.inference_mode()
    def classify(self, tokens: torch.Tensor) -> torch.Tensor:
        """The head's probability of each of ``labels`` for a clip's tokens, float32."""
        if self.pooler is None:
            raise RuntimeError("loaded without the head (labels=False)")
        logits = self.classifier(self.pooler(tokens[None]))[0]
        return logits.float().softmax(-1)
