# Changelog

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
