# Jax re-implementation of EDM

Written by Kaiming He.

### Introduction

Before you start, please read [He Vision Group's TPU intro repo](https://github.com/KaimingHe/resnet_jax).

### Usage

See `run_script.sh` for a dev run in local TPU VM.

See `run_remote.sh` for preparing a remote job in a remote multi-node TPU VM. Then run `run_staging.sh` to kick off.

### Results

FID: running for 4000 epochs gives 2.1~2.3 FID with N=18 sampling steps. As a reference, running the original Pytorch/GPU code has 2.04 FID, and 1.97 was reported in the paper (unconditional).

Time: set `config.fid.fid_per_epoch` to a bigger number (200 or 500) to reduce FID evaluation frequency. If FID evaluation is not much, training for 4000 epochs should take ~12 hours in a v3-32 VM. Training for 1000 epochs (~3 hours) should give ~2.6 FID.

### Known issues

EDM turns dropout on even at inference time. My code turns dropout off at inference time. This seems to have negligible difference.
