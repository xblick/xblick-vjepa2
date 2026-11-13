"""xblick-vjepa2: V-JEPA 2's video features (model.py) of a camera's colour stream, and the actions an attentive
head reads from them (Meta's Something-Something v2 head by default).

    xblick-vjepa2 --camera robot/camera [--color <camera>/color] [--node vjepa2] \\
                      [--frames 16] [--fps 8] [--checkpoint ssv2 | <directory>] [--top 5] [--no-labels] \\
                      [--local shm|cuda]
                      [--connect ...] [--listen ...] [--config FILE]   (default: the PC's Zenoh router)

Every colour frame becomes the model's input when it comes (the centre 256 x 256 of the frame scaled to a short side
of 292, as V-JEPA 2 was trained); a clip is the --frames frames 1/--fps seconds apart that end at the newest one
(clip.py: by their stamps, whatever the camera's rate). For the newest clip, as often as the GPU gets to it and at
most --max-hz clips a second (the clip's newest stamp in a later 1/max-hz period than the last clip's; every frame
still becomes an input), it publishes, stamped as the clip's newest frame and in its camera frame:

* ``<camera>/<node>/tokens``   image stream, 16FC1: the encoder's last tokens of the clip as an N x 1024 matrix (rows
  the tokens, columns their features; float16): N = frames / 2 x 16 x 16 (2048 for 16 frames), in the order time,
  row, column, oldest first: the input of an attentive head (transformers' VJEPA2AttentivePooler takes them as they
  are). Computed for every clip; published while somebody subscribes: on the GPU (``cuda`` transport, Durchblick's
  CUDA ring) for readers on the same host, else ``raw`` (4 MB a clip)
* ``<camera>/<node>/actions``  vision_msgs/Classification: the head's --top classes, best first, each its label as
  class_id and its probability (softmax over all classes) as score; not with --no-labels

Clips per second, the time from the newest frame to the results and how many frames could not make a clip (a gap in
the stream, or not yet a clip's time since the start) go to ``<camera>/diagnostics`` (status ``vjepa2``) once a second.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time

import xblick as db
from xblick import mps, node, tensors, topics
from xblick.camera import check_local
from xblick.history import FrameHistory
from xblick.session import add_arguments, open_from_args

from .clip import Clip

KEEP_FRAMES = 30  # frames kept until made inputs: a second at 30 frames/s while the encoder works


def clip_due(newest: int, clipped: int, period_ns: int | None) -> bool:
    """Whether the clip ending at stamp ``newest`` is encoded under --max-hz: its stamp in a later 1/max-hz period
    (counted from the epoch, as Durchblick's subscriptions do) than ``clipped``, the last encoded clip's (-1: none
    yet); always without a rate (``period_ns`` None)."""
    return period_ns is None or clipped < 0 or newest // period_ns > clipped // period_ns


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="xblick-vjepa2", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--camera", required=True, help="the camera's base, e.g. robot/camera")
    parser.add_argument("--color", default=None, help="the colour stream (default <camera>/color)")
    parser.add_argument("--node", default="vjepa2", help="the results' part of the topic: <camera>/<node>/...")
    parser.add_argument("--frames", type=int, default=16,
                        help="frames per clip, even (two frames make a token's time step); the SSv2 head was trained "
                             "on 16")
    parser.add_argument("--fps", type=float, default=8.0, help="frames per second of the clip (16 at 8: 2 s)")
    parser.add_argument("--checkpoint", default="ssv2",
                        help="a VJEPA2ForVideoClassification: ssv2 (Meta's Something-Something v2 head, in weights/ "
                             "from tools/setup.sh) or a directory with its config.json and model.safetensors")
    parser.add_argument("--top", type=int, default=5, help="classes per Classification")
    parser.add_argument("--no-labels", action="store_true", help="only the tokens, no head")
    parser.add_argument("--device", default="cuda")
    add_arguments(parser)  # --connect, --listen, --config; default: the PC's Zenoh router
    # --max-hz: clips a second (the input takes every frame); every clip by default
    # --local: auto (the CUDA ring on this GPU, else shared memory), shm or cuda
    node.add_arguments(parser, local=("auto", "shm", "cuda"))
    node.Parameters.add_arguments(parser)  # --param NAME:=VALUE, --params-file, --dump-params, --params-read-only
    args = parser.parse_args(argv)
    mps.check()  # with CUDA_MPS_PIPE_DIRECTORY (MPS meant) its daemon has to run: else stop here
    # the options as parameters (the terminal interface shows them): read-only, but top (classes per Classification)
    session = open_from_args(args)
    params = node.Parameters(session, args.node, topics.check_name(args.camera))
    params.configure(args)
    params.declare_options(parser, args, live={"top": {"range": (1, 400, 1), "unit": "classes"}},
                           leave_out=("camera", "node"))
    params.check_overrides()
    if args.frames < 2 or args.frames % 2:
        parser.error("--frames: an even number, at least 2")
    if args.fps <= 0 or args.top < 1:
        parser.error("--fps: positive; --top: at least 1")
    camera = topics.check_name(args.camera)
    color_base = args.color or f"{camera}/color"

    node.start("xblick-vjepa2", max_hz=args.max_hz, local=args.local)

    import torch  # noqa: PLC0415 - with the model, after the arguments are checked

    from .model import VJepa2  # noqa: PLC0415 - loads transformers

    device = torch.device(args.device)
    model = VJepa2(args.checkpoint, device, labels=not args.no_labels)
    if model.labels and args.frames != model.frames_trained:
        print(f"xblick-vjepa2: the head was trained on clips of {model.frames_trained} frames, these have "
              f"{args.frames}", flush=True)
    wake = threading.Event()
    # every frame: a clip's frames are 1/--fps apart, and --max-hz limits the clips, not their frames
    color = db.ImageSubscriber(session, color_base, transport="auto", local=args.local, device=device)
    clip_period_ns = None if args.max_hz is None else round(1e9 / args.max_hz)
    history = FrameHistory(color, KEEP_FRAMES, wake.set)
    clip = Clip(args.frames, round(1e9 / args.fps))
    tokens_pub = db.CameraPublisher(session, f"{camera}/{args.node}/tokens", encoding="16FC1", frame_id="",
                                    source="vjepa2", cuda_device=device.index if device.index is not None else 0,
                                    codec=None)  # half floats: no compressed form
    actions_pub = (db.DetectionPublisher(session, topics.detections(camera, args.node, "actions"), db.Classification)
                   if model.labels else None)
    report = node.Reporter(session, camera, args.node, "xblick-vjepa2", max_hz=args.max_hz)
    report.watch(color)
    outputs = "tokens" + (",actions" if actions_pub is not None else "")
    print(f"xblick-vjepa2: {color_base} -> {camera}/{args.node}/{{{outputs}}}: clips of {args.frames} frames at "
          f"{args.fps:g} frames/s ({model.tokens_of(args.frames)} tokens of {model.width})"
          + (f", {len(model.labels)} classes ({args.checkpoint})" if model.labels else ""), flush=True)
    counts = report.counts
    prepared, encoded = -1, -1  # the newest stamps made inputs, and looked at for a clip
    clipped = -1  # the newest stamp of the last clip encoded (for --max-hz)
    frame_id = ""  # the camera frame of the newest input

    try:
        while True:
            wake.wait(0.05)
            wake.clear()
            history.check()
            with history.lock:
                new = sorted((s, f) for s, f in history.items.items() if s > prepared)
            for stamp, frame in new:
                check_local(color, args.local)
                clip.put(stamp, model.prepare(tensors.rgb(frame, device)))
                prepared, frame_id = stamp, frame.header.frame_id
                counts["frames"] += 1
            newest = clip.newest
            if newest is not None and newest != encoded and not clip_due(newest, clipped, clip_period_ns):
                counts["skipped_max_hz"] += 1  # in the last clip's 1/--max-hz period
                encoded = newest
            elif newest is not None and newest != encoded:
                encoded = newest
                taken = clip.take()
                if taken is None:
                    counts["no_clip"] += 1
                else:
                    _, inputs = taken
                    tokens = model.encode(inputs)
                    if actions_pub is not None:
                        probabilities, classes = model.classify(tokens).topk(params["top"])
                        header = db.Header(db.Time.from_ns(newest), frame_id)
                        actions_pub.publish(db.Classification(header, [
                            db.ObjectHypothesis(model.labels[c], p)
                            for p, c in zip(probabilities.tolist(), classes.tolist(), strict=True)]))
                    if tokens_pub.subscribed:
                        tokens_pub.frame_id = frame_id
                        # ordered on this thread's stream after the encoder: publish copies it once it is complete
                        tokens_pub.publish(tokens.half(), stamp_ns=newest, stream=torch.cuda.current_stream(device))
                    torch.cuda.current_stream(device).synchronize()  # one clip at a time on the GPU, not a queue
                    counts["clips"] += 1
                    clipped = newest
                    counts["latency"] += time.time() - newest / 1e9
            if report.due():
                d, seconds = report.take()
                ms = d["latency"] / max(d["clips"], 1) * 1e3
                problems = ([f"{d['no_clip']} frames without a clip (a gap, or starting)"] if d["no_clip"] else []) + (
                    [f"{clip.restarts} restarts of the stream"] if clip.restarts else [])
                waiting = None if d["clips"] else f"{color_base}: {color.describe()}"
                level, message = node.state(f"{d['clips'] / seconds:.2f} clips/s", problems, waiting)
                report.publish(level, message, {"clips/s": d["clips"] / seconds, "frames/s": d["frames"] / seconds,
                                                "ms from frame to results": ms, "frames without a clip": d["no_clip"],
                                                "restarts": clip.restarts,
                                                "clips skipped for max-hz": d["skipped_max_hz"]},
                               log_extra=f", {ms:.0f} ms from the newest frame to the results")
    except KeyboardInterrupt:
        pass
    finally:
        history.close()  # and its subscriber
        for closable in (tokens_pub, actions_pub, report):
            if closable is not None:
                closable.close()
        params.close()
        session.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
