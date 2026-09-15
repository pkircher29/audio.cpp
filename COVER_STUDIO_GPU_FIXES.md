# Cover Studio GPU fixes

This is a compatibility fork of [0xShug0/audio.cpp](https://github.com/0xShug0/audio.cpp),
which provides the native inference engine for [Cover Studio](https://github.com/pkircher29/cover-studio).
All upstream authorship, history, licenses, and vendored notices are retained.
The music model is [YuE2 from the YuE project](https://github.com/multimodal-art-projection/YuE).

Based on upstream `87544b5` (Optimize Yue2 AR decode cache path).
Changes made on 2026-09-15:

1. Copy offset RoPE position views into aligned storage on Vulkan. The second
   classifier-free-guidance row previously bound an int32 view at byte offset 4,
   triggering the Vulkan alignment assertion. CPU and CUDA behavior is unchanged.
2. Cap COL2IM_1D's X dispatch to the physical device limit. Its shader already
   loops across remaining output values; the prior unbounded dispatch aborted
   in the final audio decoder for sufficiently large tensors.
3. Add a CPU-reference RoPE test and an independent scatter-add reference test
   covering 16,800,008 COL2IM_1D output values (beyond 65,535 x 256 threads).

## Validation

Windows Vulkan build, Intel Arc B580 and NVIDIA GTX 1650 SUPER:

```text
rope_position_view_test --vulkan: PASS on both cards
col2im_dispatch_test --vulkan: PASS on both cards
rope_position_view_test (CPU): PASS
```

Real YuE2 generation on Arc B580 (12 GB), Q4_0 model and F16 VAE:

- 10.239-second stereo WAV at 48 kHz, about 10 seconds elapsed.
- Two 59.959-second stereo WAVs, 27.4 and 26.7 seconds elapsed (4 ODE steps).
- Browser cover workflow: Qwen3-ASR Vulkan transcription, SheetSage2 CPU melody
  transcription, then a 16-step melody-conditioned YuE2 generation producing
  89.919 seconds of stereo audio. Engine generation took 136 seconds; the whole
  cover stage including cold melody setup took about 218 seconds.

The Cover Studio launcher uses 64 MiB weight-metadata arenas and 128 MiB graph
metadata arenas. These are host-side no-alloc GGML contexts; model tensors still
reside on the selected backend. The upstream multi-gigabyte metadata defaults
exhausted host commit during our tests. This fork does not change those defaults;
the application sets the session options explicitly.

These checks do not establish full-model generation on a 4 GB GPU, arbitrary
song lengths, musical quality, or hardware coverage beyond the listed tests.

## Licenses

The engine code remains under its existing [Apache-2.0 license](LICENSE).
Vendored libraries retain their existing license files. Modified source files
include dated change notices. Model weights are installed separately: YuE2
weights use [CC BY-NC 4.0](https://github.com/multimodal-art-projection/YuE/blob/main/MODEL_LICENSE).
The engine's license does not relicense those weights or grant endorsement.
