# Reproducibility and reported result

## What the source release reproduces

The two public training configs carry the reference hyperparameters. Both use OpenVid64K, LoRA rank 128 and alpha 64, 49-frame 480×720 buckets, a per-device batch of four for 2B and two for 5B, a temporal drift offset of two, video and frame weighting strengths of 0.4, a frame gate scale of 0.07, and checkpoint saves every 500 optimizer steps. The 2B recipe uses eight GPUs for 2,000 configured steps; the 5B recipe uses sixteen GPUs for 2,000 configured steps. The video-level rank is recomputed from the *current* global distributed batch, including ties, at each step.

The released source does not contain the original dataset, model weights, checkpoints, full evaluation outputs, or private cluster launch configuration. Six selected generated examples are included under `assets/examples/`. Paths are supplied at runtime. Consequently, a new run can differ because of model checkpoint revision, video selection/cropping, GPU topology, stochastic data order, PyTorch/CUDA kernels, or evaluator revision.

## VideoPhy measurement

The completed 5B step-500 evaluation used the included 344-prompt VideoPhy split, the native 49-frame, 8-fps generation path, 50 inference steps, classifier-free guidance scale 6.0, motion-guidance strength 15.0, guidance-step ratio 0.8, LoRA scale 64/128, and seed 42. Generation ran on eight 80 GB GPUs with two parallel workers per GPU. The completed manifest had 344 unique nonempty MP4 files and the evaluator produced 688 task scores: one SA and one PC task per prompt.

| Metric | Value |
|---|---:|
| Semantic adherence (SA) | 0.6541 |
| Physical commonsense (PC) | 0.3866 |

These values came from a single completed checkpoint and are preliminary. The package does not include the MP4 files or raw evaluator output, so the summary is not independently auditable from this ZIP alone. A public checkpoint and immutable evaluator artifact would be needed for full numerical replication. No 2B score, multi-seed interval, or VBench result is claimed here.

## Verification checklist for a new run

1. Record the base model revision, dataset CSV hash, and local video-corpus version.
2. Save `resolved_config.yaml`, the runtime dataset JSON, software versions, GPU model, global world size, and random seeds.
3. Confirm `PHYSTRATA_DRIFT=1` and the correct global batch (32 for both 2B and 5B).
4. Record checkpoint step and hashes of `pytorch_lora_weights.safetensors` and `predictor.safetensors`.
5. Use the same VideoPhy prompt CSV, scorer checkpoint, and generation settings; confirm 344 distinct nonempty videos and 688 score rows before quoting overall SA/PC.

## Ablation boundaries

The weighting is restricted to the drift-loss aggregation. A scientifically controlled comparison should keep the base model, dataset order, optimizer, LoRA configuration, predictor objective, sampling schedule, and inference guidance fixed. To isolate the video and frame effects, compare video-only, frame-only, and both-enabled settings while preserving the same global batch size. In the current source, the `PHYSTRATA_DRIFT` environment switch controls the complete weighting path; modifying coefficients for ablations requires an explicit code/config change and a separately named run.
