# xblick-vjepa2

[![xblick core](https://img.shields.io/pypi/v/xblick?label=xblick%20core&color=blue)](https://github.com/xblick/xblick)
[![tests](https://github.com/xblick/xblick-vjepa2/actions/workflows/tests.yml/badge.svg)](https://github.com/xblick/xblick-vjepa2/actions/workflows/tests.yml)
[![licence](https://img.shields.io/badge/licence-Apache--2.0-blue)](LICENSE)
[![device](https://img.shields.io/badge/needs-NVIDIA%20GPU-76b900)](README.md)

This [Durchblick](https://github.com/xblick/xblick) node computes video features for the newest
clip of a camera's colour stream, and it names the actions that a classification head reads from them. The encoder
needs 17 ms per clip of 16 frames, and the node produced a clip for every frame of a camera at 15 frames per second.

The node uses Meta's [V-JEPA 2](https://github.com/facebookresearch/vjepa2) (2025). V-JEPA 2 is a video encoder that
Meta trained without labels on over a million hours of video. It learned to predict masked parts of a clip in
feature space. The encoder turns a clip into *tokens*, which are feature vectors for small pieces of the clip.

The tokens are the main output of the node. Every V-JEPA 2 head reads them, so a head of your own that you trained
on clips of your robot reads them as they are. The node also publishes the actions of Meta's head for
Something-Something v2 (SSv2), a dataset of 174 classes of hand-object actions.

## What it publishes

The node publishes for the newest clip, as often as the GPU finishes one. Each message has the stamp of the newest
frame of the clip and the camera frame of that frame.

| Topic | Message type | Content |
| --- | --- | --- |
| `<camera>/vjepa2/tokens` | image stream, encoding `16FC1` | the last tokens of the encoder as a matrix of N x 1024 float16 values, one row per token. N is 2048 for a clip of 16 frames. |
| `<camera>/vjepa2/actions` | `vision_msgs/Classification` | the `--top` classes of the head, the best first. `class_id` is the label, and `score` is its probability. |
| `<camera>/diagnostics` | `diagnostic_msgs/DiagnosticArray` | status `vjepa2` once per second: clips and frames per second, ms from the newest frame to the results, frames without a clip, restarts, clips skipped for `--max-hz` |

The node publishes the tokens only while a subscriber exists. It does not publish actions with `--no-labels`.
`xblick rerun` shows the actions as a text log. `--node` changes the `vjepa2` part of the topics.

The node reads `<camera>/color` in the encoding `rgb8`, `rgba8`, `bgr8`, `bgra8` or `mono8`.
[How the node works](docs/design.md) describes the tokens and how a head reads them.

## Requirements

| What | Requirement |
| --- | --- |
| GPU | an NVIDIA GPU with driver 580 or newer (PyTorch for CUDA 13) |
| VRAM | 1.4 GiB with clips of 16 frames at 8 frames per second (the node alone, 1280 x 720 at 30 frames per second, measured 2026-11-02) |
| Durchblick | 0.5 or newer, below 0.6 (Python package `xblick`) |
| Python | 3.12, which uv installs with the node |
| Tools for the install | git and uv (step 1 of the installation installs them) |
| Disk | 1.4 GB for the checkpoint |

Model licences:

| Part | Licence |
| --- | --- |
| Model code, from the `transformers` package | Apache 2.0. It is a dependency. |
| Checkpoint `facebook/vjepa2-vitl-fpc16-256-ssv2` | MIT, as its Hugging Face model card states. It is a download. |

The node downloads the checkpoint and prints these terms first. This repository contains no copy of the code or the
checkpoint.

## Install

1. Install the prerequisites. `nvidia-smi` prints the GPU and the driver version, which must be 580 or newer. On
   Ubuntu, `sudo apt install nvidia-driver-580` and a restart install that driver. After the installation of uv, open
   a new terminal, so that `uv` is on `PATH`.

   ```bash
   nvidia-smi
   sudo apt install git
   wget -qO- https://astral.sh/uv/install.sh | sh
   ```

2. Install the node and download its weights:

   ```bash
   uv tool install git+https://github.com/xblick/xblick-vjepa2@v0.2.6
   xblick-vjepa2 --download-only
   ```

   uv puts `xblick-vjepa2` on `PATH`, in an environment of its own, with PyTorch for CUDA 13 (about 3 GB to download,
   6 GB on disk) and Python 3.12. It resolves the dependencies afresh within the bounds of `pyproject.toml`:
   `uv.lock` applies only to a clone. `xblick-vjepa2 --download-only` prints the model's terms and downloads the
   SSv2 checkpoint (1.4 GB) from Hugging Face at a fixed revision into `~/.cache/xblick/vjepa2/ssv2`.

   `uv tool upgrade xblick-vjepa2` updates the dependencies within the bounds of the installed version. A newer
   version is the same install command with its tag. `uv tool uninstall xblick-vjepa2` removes the node. The
   checkpoint stays in the cache directory.

3. Install the core's command `xblick` with the viewer, video files and webcams. uv puts it on `PATH`. A second
   `uv tool install` of xblick with other extras replaces the first and removes what only the first had, so every
   xblick node names this one line:

   ```bash
   uv tool install "xblick[rerun,video,cam]>=0.5,<0.6"
   ```

4. Start the node on a video, without a camera. Any short video of a hand at work works, for example one from a
   phone of a hand that picks something up. Run each command in its own terminal. If the PC already runs a Zenoh
   router (`xblick router` or `zenohd`), leave out the first one.

   ```bash
   xblick router                                               # the Zenoh router of this PC
   xblick play ~/Videos/clip.mp4 --camera demo/video --loop    # the video as the camera demo/video
   xblick-vjepa2 --camera demo/video
   xblick rerun                                                # the viewer
   ```

   The viewer shows the video and beside it a text log with the five actions of Something-Something v2 that
   V-JEPA 2 finds most likely in the last two seconds, such as "Picking [something] up", with their scores.

### From a clone

For development, and for the versions that `uv.lock` fixes:

```bash
git clone https://github.com/xblick/xblick-vjepa2.git
cd xblick-vjepa2
tools/setup.sh
```

When git or uv is missing, the script stops and prints the command that installs it. Without an NVIDIA GPU with
driver 580 or newer it says so and goes on. `tools/setup.sh --help` describes it. In a clone the commands are in
`.venv/bin/`.

The script creates `.venv` exactly as `uv.lock` states, with PyTorch for CUDA 13, transformers and Durchblick from
PyPI. Then it runs `xblick-vjepa2 --download-only`, which downloads the SSv2 checkpoint from Hugging Face at a
fixed revision into `~/.cache/xblick/vjepa2/ssv2`. It downloads only what is missing.

### Downloads at the start

The node itself also downloads a missing checkpoint at its start, before it loads the model. It prints the model's
terms, the files and their size first. A preparation script downloads without a start with
`xblick-vjepa2 --download-only` (`.venv/bin/xblick-vjepa2` in a clone).

`--download-only` needs no camera, no Zenoh router and no GPU, and it exits with 0 when the weights are complete. A
machine that must never start a download on its own, such as a robot, starts the node with `--no-download`. The
node then ends at once when a file is missing, and it names the file.

### Where the weights are

The node keeps its weights in its cache directory, `~/.cache/xblick/vjepa2/`. `XBLICK_HOME=/dir` moves it to `/dir/cache/vjepa2/`,
and `XDG_CACHE_HOME` moves it as well. `--weights-dir DIR` names another directory. The place is the same for an
install with uv and for a clone. [Where files are](https://github.com/xblick/xblick/blob/main/docs/reference/files.md) lists the places, and it shows how to copy the cache to a
PC without internet.

Up to version 0.2.2, the weights were in the checkout's `weights/`. The node does not read them there. At its start it
prints the command that moves them. In the checkout it is:

```bash
mkdir -p ~/.cache/xblick && mv weights ~/.cache/xblick/vjepa2
```

## Run

Start a Zenoh router (`xblick router` or `zenohd`) on the PC, then start the node. The camera name is an example.
In a clone the commands are in `.venv/bin/`.

```bash
xblick-vjepa2 --camera test/camera
```

Look at the result in Durchblick's viewer. It shows the actions as a text log beside the images.

```bash
xblick rerun --spawn
```

Two variants publish only the tokens, or use a head of your own. Such a head is a `VJEPA2ForVideoClassification`
that you saved with `save_pretrained`.

```bash
xblick-vjepa2 --camera test/camera --no-labels
xblick-vjepa2 --camera test/camera --checkpoint /path/to/our_head
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--camera CAMERA` | required | the base name of the camera |
| `--frames N` | 16 | the frames per clip, an even number. Meta trained the SSv2 head on 16. |
| `--fps HZ` | 8 | the frames per second inside the clip. 16 frames at 8 give 2 s. |
| `--checkpoint NAME` | `ssv2` | `ssv2` or a directory with `config.json` and `model.safetensors` |
| `--top N` | 5 | the classes per `Classification`. Live: it changes while the node runs. |
| `--no-labels` | off | compute only the tokens and load no head |
| `--max-hz HZ` | every clip | the most clips per second. `0` or `none` also means every clip. |
| `--local` | `auto` | how frames of a camera on this host reach the node: `auto`, `shm` or `cuda` |
| `--color TOPIC` | `<camera>/color` | the colour stream |
| `--node NAME` | `vjepa2` | the node's part of the topics |
| `--device` | `cuda` | the GPU: `cuda` is GPU 0, `cuda:N` is GPU N. A GPU that cannot be used ends the node with one line. |
| `--no-download` | off | end at once when a weight file is missing. The node downloads nothing. |
| `--download-only` | off | download the missing weights, check them and end. `--camera` may be left out. |
| `--weights-dir DIR` | the cache directory | where the weights are and go, in place of `~/.cache/xblick/vjepa2/` |

[How the node works](docs/design.md) explains the options in detail.

Every option is also a parameter of the node `<camera>/vjepa2`, which Durchblick's terminal interface (`xblick`, tab Parameters) and `xblick param` show. `top` changes while the node runs. The other options apply at the next start, where `--param NAME:=VALUE` or `--params-file FILE` set them. `--download-only` is a mode of the command line and has no parameter.

## Why it is fast

The encoder needs 17 ms per clip of 16 frames on an RTX PRO 6000, and the head needs about 1 ms.

| Choice | Effect |
| --- | --- |
| The model computes in bfloat16, a 16-bit number format that GPUs compute faster than 32-bit floats. | part of the 17 ms |
| The model uses PyTorch's scaled-dot-product attention. | part of the 17 ms |
| The node prepares each frame once, on the GPU, when the frame arrives. | a clip only stacks 16 prepared frames |
| The tokens stay on the GPU for readers on the same host. | no 4 MB copy per clip |

[Speed](docs/performance.md) has the measurements and their setup.

## Limits

- SSv2 shows hands with objects, filmed close up. The view of a robot on a person farther away differs from the
  training data, so the head gives odd labels there.
- The default of 8 frames per second is our choice. It may differ from the frame rate that Meta trained the SSv2
  head with.
- The node crops the centre of the image. The model does not see the sides of a wide image.
- A gap in the stream gives no clip. The diagnostics count such frames.

## Documentation

- [How the node works](docs/design.md): the encoder, the clip, the tokens, a head that reads them, the options,
  use from Python, tests
- [Speed](docs/performance.md): the timings and the live measurement
- Durchblick: [streams and topics](https://github.com/xblick/xblick/blob/main/docs/concepts/streams-and-topics.md),
  [frames on the GPU](https://github.com/xblick/xblick/blob/main/docs/concepts/gpu-frames.md),
  [rings](https://github.com/xblick/xblick/blob/main/docs/concepts/rings.md),
  [transports](https://github.com/xblick/xblick/blob/main/docs/concepts/transports.md),
  [sessions and routers](https://github.com/xblick/xblick/blob/main/docs/concepts/sessions.md)

## Licence and credits

V-JEPA 2 is by Meta ([facebookresearch/vjepa2](https://github.com/facebookresearch/vjepa2)). The model code is the
one of Hugging Face's transformers. "Requirements" lists the licences.

The code of this repository is under the Apache License 2.0, see [LICENSE](LICENSE). [NOTICE](NOTICE) names the
copyright holder and the parts of others.

## Citation

This node uses V-JEPA 2 (facebook/vjepa2-vitl-fpc16-256-ssv2 on Hugging Face). If you use it in research, cite their
work. The entries below are copied from [its
README](https://huggingface.co/facebook/vjepa2-vitl-fpc16-256-ssv2/blob/4aa02df83918538fc21cfaf576382fa20e489a80/README.md):

```bibtex
@techreport{assran2025vjepa2,
  title={V-JEPA~2: Self-Supervised Video Models Enable Understanding, Prediction and Planning},
  author={Assran, Mahmoud and Bardes, Adrien and Fan, David and Garrido, Quentin and Howes, Russell and
  Komeili, Mojtaba and Muckley, Matthew and Rizvi, Ammar and Roberts, Claire and Sinha, Koustuv and Zholus, Artem and
  Arnaud, Sergio and Gejji, Abha and Martin, Ada and Robert Hogan, Francois and Dugas, Daniel and
  Bojanowski, Piotr and Khalidov, Vasil and Labatut, Patrick and Massa, Francisco and Szafraniec, Marc and
  Krishnakumar, Kapil and Li, Yong and Ma, Xiaodong and Chandar, Sarath and Meier, Franziska and LeCun, Yann and
  Rabbat, Michael and Ballas, Nicolas},
  institution={FAIR at Meta},
  year={2025}
}
```
