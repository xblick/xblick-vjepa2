# AGENTS.md

xblick-vjepa2: Meta's V-JEPA 2 ViT-L video encoder as a Durchblick node. Publishes the encoder's tokens for the
newest clip of a colour stream (the main output: heads of our own read them) and the top classes of the
Something-Something v2 head. Model code is transformers' (`VJEPA2ForVideoClassification`), a dependency. Version
0.2.0, tested against transformers' pipeline and in a simulation, not on a robot. Human documentation:
`README.md` (topics, options, limits), `docs/design.md`, `docs/performance.md`. Transport-wide traps
(Zenoh sessions, CUDA rings, looped playback, hanging tests) are in the core's `AGENTS.md`
(github.com/xblick/xblick). Style of all documentation: the core's `docs/project/writing-docs.md`.

## Commands

```bash
tools/setup.sh                                     # uv sync --frozen, SSv2 checkpoint (1.4 GB) into weights/ssv2
.venv/bin/python -m pytest -q                      # 11 tests; test_model.py needs a CUDA device and weights/ssv2
.venv/bin/python -m pytest -q tests/test_clip.py   # CPU only: which frames make a clip
.venv/bin/ruff check src tests                     # lint; 120 columns, rules in pyproject.toml
.venv/bin/xblick-vjepa2 --camera test/camera   # needs zenohd on the PC
```

- Console script: `xblick-vjepa2`. `--no-labels`: tokens only. `--checkpoint DIR`: a head of our own.
- `tools/setup.sh` needs uv (`UV` names it) and git. It downloads only what is missing.
- Durations: not recorded.
- torchvision is a dev dependency only (transformers' video processor in the tests).
- Without a camera: a simulation that publishes a camera, or `xblick play <recording.mcap> --loop`.
  Input is `<camera>/color` only, no other node needed.
- `xblick rerun --spawn` shows the actions as a text log.

## Rules

- Match transformers exactly. `model.prepare` reimplements `VJEPA2VideoProcessor` on the GPU. `tests/test_model.py`
  checks inputs for four frame sizes and the probabilities against `VJEPA2ForVideoClassification`. Keep those tests
  green after any change to the preprocessing or the model call.
- Work offline. Checkpoints come from `model.CHECKPOINTS` (repository, fixed revision) through `tools/setup.sh` into
  `weights/<name>`. `--checkpoint` takes such a name or a directory saved with `save_pretrained`.
- Do not reorder the tokens for the transport.
- Durchblick pin: `xblick>=0.5.0,<0.6` (the core's Python package).
- 120 columns. Docstrings say what, comments say why.
- Update `README.md` and `docs/` with every change of behaviour.
- Run GPU tests and live checks in the background with `timeout`. Stop them by PID.

## Traps

| Trap | What happens | Fix |
| --- | --- | --- |
| Centre crop rounds down | transformers' crop is `(w - 256) // 2` (odd margins floor). torchvision rounds differently. | Keep the floor. |
| Resize | Antialiased bilinear on uint8, rounded to integers, short side `256 * 256 / 224 = 292`. A wide image's sides are not seen. | Keep it, `test_model.py` compares with the processor. |
| Token order | Time, row, column. One token per two frames (tubelet) and 16 x 16 pixels: N = frames / 2 x 16 x 16 (2048 at 16 frames), as `VJEPA2AttentivePooler` takes them. | Never reorder. |
| Clips by stamp (`clip.py`) | Each clip frame is the camera frame nearest its time, whatever the camera's rate. A frame may be used twice. A frame further than one period from its time is a gap and gives no clip. | |
| Stream restart | A stamp older than the newest by more than a clip restarts the clip. | Counted as `restarts` in diagnostics. |
| Encoder is the pretrained one | The SSv2 checkpoint's encoder weights are bit-identical to `vjepa2-vitl-fpc64-256` (Meta trains only heads). The pretraining predictor is dropped. bfloat16, SDPA attention. | |
| Not verified | The frame rate Meta's SSv2 head was trained with (8 frames/s is a choice here). | |
| Checkpoint licence | MIT, per the Hugging Face model card of `facebook/vjepa2-vitl-fpc16-256-ssv2`. | |
| SSv2 labels | SSv2 is close-up hand-object video. A robot's view gives odd labels. | |
| Tokens transport | Published only while somebody subscribes. Same host gets the CUDA ring (`local="cuda"`), else raw (4 MB a clip). | |
| CUDA ring in tests | The process that created a CUDA ring cannot open it. | Tests read it from a `spawn`ed child. |
| `--frames` | Must be even and at least 2. A note is printed when the head was trained on another count. | |
| MPS check | With `CUDA_MPS_PIPE_DIRECTORY` set, `mps.check()` stops the node when the daemon does not answer. | Start MPS first or unset the variable. |
| Zenoh sessions (core) | A session with only `--connect` does not listen, and the router does not route between such sessions. | Start nodes without endpoints. They find the PC's router. |

## Layout

| Path | Content |
| --- | --- |
| `src/xblick_vjepa2/model.py` | `VJepa2`: `prepare` (per frame), `encode` (per clip), `classify`; `CHECKPOINTS` |
| `src/xblick_vjepa2/clip.py` | `pick`/`Clip`: which frames make the clip, by stamp |
| `src/xblick_vjepa2/node.py` | the node: subscription, clip, publishing tokens/actions, diagnostics |
| `weights/<name>/` | `config.json`, `model.safetensors` (`tools/setup.sh`). Not in git. |
| `tests/` | `test_clip.py` (CPU), `test_model.py` (GPU) |
| `tools/setup.sh` | venv and checkpoint |

## Defaults

| What | Value |
| --- | --- |
| Input | `<camera>/color` (`--color`): `rgb8`, `rgba8`, `bgr8`, `bgra8`, `mono8` |
| Tokens | `<camera>/vjepa2/tokens`, image stream `16FC1`, 2048 x 1024 float16, order time-row-column, oldest first |
| Actions | `<camera>/vjepa2/actions`, `vision_msgs/Classification`, `--top` 5 |
| Diagnostics | `<camera>/diagnostics`, status `vjepa2`, once per second |
| Clip | `--frames` 16, `--fps` 8 (2 s), picked by stamp |
| `--checkpoint` | `ssv2` (`facebook/vjepa2-vitl-fpc16-256-ssv2`, fixed revision in `model.CHECKPOINTS`) |
| `--max-hz` | none (every clip) |
| `--local` | `auto` (`shm`, `cuda`) |
| `--device` | `cuda` |
| `KEEP_FRAMES` | 30 |
| Environment | `UV` (setup), `CUDA_MPS_PIPE_DIRECTORY` (MPS) |
