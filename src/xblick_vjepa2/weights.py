"""The node's weights: which files it reads, where they come from, and their download.

This module is the one list of them: the node's start and ``tools/setup.sh`` (through ``xblick-vjepa2
--download-only``) both use it. ``STORE.ensure`` touches neither the disk nor the network when every file is there. A
missing file is downloaded under a temporary name in its directory (``<name>.part``), checked and only then renamed,
all under a file lock, so that two nodes started at once download once and never read half a file.

* ``CHECKPOINTS``: name -> (Hugging Face repository, revision, {file: (size in bytes, SHA-256)}), in ``<name>/`` of
  the node's cache directory (``~/.cache/xblick/vjepa2``, ``--weights-dir``); the SHA-256 of an LFS file is Hugging
  Face's ``lfs.oid`` at the revision, that of a small file the hash of the file at the revision (both checked against
  a copy downloaded before)
* ``needed(checkpoint)``: the files a start with ``--checkpoint`` reads (none for a directory of one's own), their paths
  relative to the store's root
* ``store_for(parser, args)``: ``STORE`` at ``--weights-dir``; ``options(args)``: the options that select the weights,
  for the command the lines name

The download itself (lock, ``.part`` files, checks, progress, ``--download-only``/``--no-download``) is the core's
``xblick.weights``; this module keeps the node's files, its terms notice and its ``STORE``. The checkout's
``weights/`` is the node's former place: weights still there are not used, and the store prints the command that
moves them.
"""

from __future__ import annotations

from pathlib import Path

from xblick.weights import HuggingFace, Store, Weight

NODE = "vjepa2"  # the short name: the cache directory ~/.cache/xblick/vjepa2

PROG = "xblick-vjepa2"
REQUIRED = ()  # (path, what): code from tools/setup.sh that must be there first; none here
REPO = Path(__file__).resolve().parents[2]
FORMER = REPO / "weights"  # where the weights were before 0.2.3 (the checkout): not used, the store names the move
CHECKPOINTS = {  # name -> (Hugging Face repository, revision, {file: (size, SHA-256)}): into <root>/<name>
    "ssv2": ("facebook/vjepa2-vitl-fpc16-256-ssv2", "4aa02df83918538fc21cfaf576382fa20e489a80", {
        "config.json": (14_953, "9ea266370f726ba85ba03080abab9632233d0742a084f0106d90d909772e4aab"),
        "model.safetensors": (1_502_018_592, "35aeab96e8513d9cc60ee08c1642dd121c86452c926ce3599b3317896a320396")}),
}
CHECKPOINT_FILES = ("config.json", "model.safetensors")
TERMS = """\
V-JEPA 2 is the work of Meta, under its terms, and you are the one who uses it:
  * The checkpoint facebook/vjepa2-vitl-fpc16-256-ssv2 (Hugging Face) is under the MIT License, as its Hugging Face
    model card states."""


def needed(checkpoint: str) -> list[Weight]:
    """The files of ``--checkpoint``: those of a name in CHECKPOINTS, none for a directory (the user's own)."""
    if checkpoint not in CHECKPOINTS:
        return []
    repo, revision, files = CHECKPOINTS[checkpoint]
    return [Weight(Path(checkpoint) / f, HuggingFace(repo, revision, f), *files[f]) for f in CHECKPOINT_FILES]


def options(args) -> list[str]:
    """The options that select the weights (and --weights-dir), for the download command the lines name."""
    return ["--checkpoint", args.checkpoint] + (["--weights-dir", args.weights_dir] if args.weights_dir else [])


def store_for(parser, args) -> Store:
    """STORE at --weights-dir (not given: the node's cache directory); a wrong --weights-dir is an argument error."""
    try:
        return STORE.at(args.weights_dir)
    except ValueError as error:
        parser.error(f"--weights-dir: {error}")


# the download (xblick.weights): the root is the node's cache directory, its lock there
STORE = Store(PROG, node=NODE, terms=TERMS, required=REQUIRED, repo=REPO, former=[FORMER])
