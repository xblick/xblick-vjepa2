# xblick-vjepa2

[![xblick core](https://img.shields.io/pypi/v/xblick?label=xblick%20core&color=blue)](https://github.com/xblick/xblick)

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
| GPU | an NVIDIA GPU with CUDA. We tested on an RTX PRO 6000. |
| VRAM | 1.4 GiB with clips of 16 frames at 8 frames per second (the node alone, 1280 x 720 at 30 frames per second, measured 2026-11-02) |
| Durchblick | 0.5 or newer, below 0.6 (Python package `xblick`) |
| Python | 3.12, which `tools/setup.sh` installs into `.venv` |
| Tools for the setup | uv and git |
| Disk | 1.4 GB for the checkpoint |

Model licences:

| Part | Licence |
| --- | --- |
| Model code, from the `transformers` package | Apache 2.0. It is a dependency. |
| Checkpoint `facebook/vjepa2-vitl-fpc16-256-ssv2` | MIT, as its Hugging Face model card states. It is a download. |

The node downloads the checkpoint and prints these terms first. This repository contains no copy of the code or the
checkpoint.

## Install

```bash
tools/setup.sh
```

The script creates `.venv` exactly as `uv.lock` states, with PyTorch for CUDA 13, transformers and Durchblick from
source. Then it runs `xblick-vjepa2 --download-only`, which downloads the SSv2 checkpoint from Hugging Face at a
fixed revision into `weights/ssv2`. It downloads only what is missing.

The node itself also downloads a missing checkpoint at its start, before it loads the model. It prints the model's
terms, the files and their size first. A preparation script downloads without a start:

```bash
.venv/bin/xblick-vjepa2 --download-only
```

`--download-only` needs no camera, no Zenoh router and no GPU, and it exits with 0 when the weights are complete. A
machine that must never start a download on its own, such as a robot, starts the node with `--no-download`. The
node then ends at once when a file is missing, and it names the file.

## Run

Start a Zenoh router (`zenohd`) on the PC, then start the node. The camera name is an example.

```bash
.venv/bin/xblick-vjepa2 --camera test/camera
```

Look at the result in Durchblick's viewer. It shows the actions as a text log beside the images.

```bash
xblick rerun --spawn
```

Two variants publish only the tokens, or use a head of your own. Such a head is a `VJEPA2ForVideoClassification`
that you saved with `save_pretrained`.

```bash
.venv/bin/xblick-vjepa2 --camera test/camera --no-labels
.venv/bin/xblick-vjepa2 --camera test/camera --checkpoint /path/to/our_head
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
| `--device` | `cuda` | the CUDA device |
| `--no-download` | off | end at once when a weight file is missing. The node downloads nothing. |
| `--download-only` | off | download the missing weights, check them and end. `--camera` may be left out. |

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

- We tested the node against the pipeline of transformers and live on a simulation. We have not tested it on a robot.
- SSv2 shows hands with objects, filmed close up. The view of a robot on a person farther away differs from the
  training data, so the head gives odd labels there.
- We have not verified the frame rate that Meta trained the SSv2 head with. The default of 8 frames per second is
  our choice.
- The node crops the centre of the image. The model does not see the sides of a wide image.
- A gap in the stream gives no clip. The diagnostics count such frames.
- We tested the download against a local server, and we fetched `config.json` from Hugging Face with it. We have
  not downloaded the 1.5 GB `model.safetensors` through the node.

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
