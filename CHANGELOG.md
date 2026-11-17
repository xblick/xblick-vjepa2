# Changelog

## Branch ros2 (not released)

The node with the core's option `--ros` (ROS 2 on `rmw_zenoh`).

### Changed

- The core comes from its unreleased branch `ros2` (`xblick @ git+https://github.com/xblick/xblick@ros2`), not from
  PyPI. `uv sync` builds its C++ part from source.

### Added

- `--ros`, `--ros-domain`, `--ros-node` and `--ros-params-writable` from the core. With `--ros` the actions and the
  tokens go to ROS 2 as well, and the parameters are a ROS node. README: section "ROS 2".
- `tests/test_ros.py`: a stand-in model (no checkpoint, a CUDA device); with `--ros` a ROS subscriber gets each output
  with a ROS type, with ROS's attachment and tokens; without it nothing of ROS is on the wire.

Compatible with earlier clients: nothing on our keys changed.

## 0.2.4 (2026-11-15)

A fixed GPU for `--device cuda`, and how to cite V-JEPA 2.

### Changed

- `--device cuda` is GPU 0, `cuda:N` GPU N. Another value, or a GPU that cannot be used, ends the node with one line.
- The start line says where the model computes (`model on cuda:0`).
- README: how to cite V-JEPA 2.

Compatible with earlier clients: nothing on the wire changed.

## 0.2.3 (2026-11-14)

- The weights moved out of the checkout into the node's cache directory, `~/.cache/xblick/vjepa2/`. `XBLICK_HOME` and
  `XDG_CACHE_HOME` move it, and `--weights-dir DIR` names another directory. Weights at the former place, the
  checkout's `weights/`, are not used. The node prints the command that moves them. In the checkout it is
  `mkdir -p ~/.cache/xblick && mv weights ~/.cache/xblick/vjepa2`.
- `--weights-dir DIR` is a new start option and a read-only parameter.
- The message about a missing file names the full download command (`xblick-vjepa2 --download-only --checkpoint ssv2`).
- The node needs `xblick>=0.5.5,<0.6`.
- `model.VJepa2` and `model.checkpoint_dir` take the weights' directory (`weights.STORE.root`, or `--weights-dir`'s).

## 0.2.2 (2026-11-14)

- The download of the weights is the core's `xblick.weights` (xblick 0.5.3). The node's own copy of that code is
  gone. `weights.py` keeps the node's files, its terms notice and its `STORE`. The options, the messages, and the
  files and their places stay as they were.
- The node needs `xblick>=0.5.3,<0.6`.
- `tests/test_weights.py` keeps the node's own checks. The core's tests cover the download itself.

## 0.2.1 (2026-11-14)

- The node downloads missing weights at its start, before the model loads. It prints the model's terms, the files
  and their size first. With all weights there it uses no network and writes no file.
- `--download-only` downloads and checks the weights, then exits 0. It needs no camera, Zenoh router or GPU, and
  `--camera` may be left out. `tools/setup.sh` calls it after `uv sync`.
- `--no-download` ends a start at once when a weight file is missing and names it. It is a read-only parameter.
- `weights.py` is the one list of the weight files. `model.CHECKPOINTS` moved there and gained the file sizes.
- A download goes to `<name>.part` under the lock `weights/.download.lock` and gets its final name after the check.
- Each weight file has its SHA-256 in `weights.py`: Hugging Face's `lfs.oid` at the fixed revision for
  `model.safetensors`, the hash of the file at that revision for `config.json`. A download is checked against it and
  against the exact size; `--download-only` also checks the files that exist.

## 0.2.0 (2026-11-13)

Every name in code, on the wire and on disk is `xblick` now. Only the prose name of the project stays Durchblick.
The node needs the core `xblick` 0.5 and does not work with programs of Durchblick 0.4 (`blick`) or older: run every
program of a setup at 0.5.

| What | 0.1.0 | 0.2.0 |
| --- | --- | --- |
| Repository | github.com/xblick/durchblick-vjepa2 | github.com/xblick/xblick-vjepa2 |
| Distribution | `durchblick-vjepa2` | `xblick-vjepa2` |
| Python package | `durchblick_vjepa2` | `xblick_vjepa2` |
| Commands | `durchblick-vjepa2` | `xblick-vjepa2` |
| Program name in log lines, the logger and reports | `durchblick-vjepa2` | `xblick-vjepa2` |
| Core dependency | `blick>=0.4,<0.5`, git main of github.com/xblick/durchblick | `xblick>=0.5.0,<0.6`, github.com/xblick/xblick at 45908e6 |
| Core imports and command | `import blick`, `blick rerun`, `blick play`, `blick param` | `import xblick`, `xblick rerun`, `xblick play`, `xblick param` |
| Links to the core's docs | github.com/xblick/durchblick/blob/main/docs/... | github.com/xblick/xblick/blob/main/docs/... |

## 0.1.0

The node as `durchblick-vjepa2` (Python package `durchblick_vjepa2`) on the core `blick` 0.4.
