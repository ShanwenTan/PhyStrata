# PhyStrata 0.1.0

Initial research source release with:

- Drift-loss weighting by global-batch video activity and gated dual-tail frame credit.
- Path-free OpenVid64K reference configs for 2B and 5B training.
- Portable single-node and two-node launcher, inference entrypoint, and VideoPhy evaluation scripts.
- Method, environment, provenance, and reproducibility documentation.
- Source checks and CPU regression tests for weights, ties, fallback behavior, and checkpoint namespace compatibility.

The archive contains no training video corpus, base model, LoRA adapter, predictor checkpoint, evaluator weights, or generated MP4. The single 5B step-500 VideoPhy measurement is documented as a preliminary reference result.
