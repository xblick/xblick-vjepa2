# Changelog

## 0.2.6 (2026-11-19)

An installation that a beginner can follow from the README alone.

### Changed

- The README's installation is a numbered path: the prerequisites with their install commands, the node with
  `uv tool install`, the core's command with one line for every node
  (`uv tool install "xblick[rerun,video,cam]>=0.5,<0.6"`), and a first start. The clone with `tools/setup.sh` stays
  the second way.
- `tools/setup.sh --help` describes the script and its options and ends. Before, it ran the setup.
- `tools/setup.sh` checks its prerequisites first. Without git or uv it stops and prints the command that installs
  them. On a PC other than Linux on x86-64, for which the core has no wheels, it stops at once.
- Before its downloads, `tools/setup.sh` lists what it downloads, with the sizes and the licences.
- `tools/setup.sh` says when it finds no NVIDIA GPU with driver 580 or newer, which the node needs to start, and
  goes on (`--download-only` needs no GPU). It refuses an unknown option; before, it ignored every option.

## 0.2.5 (2026-11-18)

The Durchblick core 0.5.14.

### Changed

- The node needs `xblick>=0.5.14,<0.6`; uv.lock fixes xblick 0.5.14 (was 0.5.5). No other package changed.
- The node declares itself on the network (core 0.5.9): its name, version, host, process and whole command line on
  `xblick/sessions/<zid>/program`. The terminal interface lists it in its Nodes and Graph tabs.
- The start line says where the node reads its frames, for example `reading the host ring: shm` (core 0.5.8).
- A `--camera` that names a stream (`cam/video2/color`) ends the node with one line that names the camera, instead of
  waiting (core 0.5.7). A router that cannot be reached ends it with one line instead of a traceback.
- A `--connect` to another PC's router in the default mode prints a notice with the three remedies (core 0.5.7).
- A parameter set whose deadline has passed when the node comes to act on it is refused, and the caller raises
  `Expired`. A set meant for another node on the same key is refused (core 0.5.13).
- The rings the node publishes record their creator, so `xblick clean --remove` removes them after the node was
  killed (core 0.5.12).
- `tests/test_weights.py`: its check for network use lets an address given as numbers resolve. The core looks up the
  hosts of `--connect` since 0.5.7, to tell whether they are this PC.

### Fixed

- A node that closed its session before its parameters no longer prints a traceback (core 0.5.7).
- Closing no longer raises "Already mutably borrowed" in the thread of the program's declaration (core 0.5.13).

Compatible with earlier clients: nothing on the wire changed.

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
