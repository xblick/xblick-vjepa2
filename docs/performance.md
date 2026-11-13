# Speed

This page reports the timings of the model and the live measurement of durchblick-vjepa2.

## Setup

Measured on 2026-11-01 on an RTX PRO 6000. The model computes in bfloat16 and uses PyTorch's scaled-dot-product
attention.

## The model

| Step | Time |
| --- | --- |
| Encoder, clip of 8 frames | 16 ms |
| Encoder, clip of 16 frames | 17 ms |
| Encoder, clip of 32 frames | 32 ms |
| Encoder, clip of 64 frames | 51 ms |
| Head | about 1 ms |

## Live

The camera was the head camera of a simulation at 15 frames per second (a turntable scene, 80 s).

| Measure | Result |
| --- | --- |
| Clips | 14.9 per second, one clip for every frame |
| Time from the capture of the newest frame to the arrival of the actions | median 38 ms, 90 %: 49 ms |

The time includes the rendering of the simulation and the transport.

The node alone uses 1.4 GiB of VRAM with clips of 16 frames at 8 frames per second. We measured this on 2026-11-02
on an RTX PRO 6000 with a camera of 1280 x 720 at 30 frames per second.

## What the head said

On the turntable, the first classes of the head were "Turning the camera left while filming [something]", "Tilting
[something] with [something] on it slightly ..." and "Pushing [something] with [something]". SSv2 shows hands close
up. Spinning objects that a robot sees are a different kind of video.

## Status and limits

- We have not measured the node on a robot.
