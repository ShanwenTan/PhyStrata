# Contributing

Thanks for considering a contribution. Keep changes focused and state which training, inference, or evaluation behavior they affect.

1. Open an issue describing the problem and a small reproducer before large changes.
2. Use a new branch and include a clear description of the expected behavior.
3. Run `python -m compileall -q finetrainers train tests eval scripts` and `python scripts/check_release.py`.
4. If changing the drift weighting, include a numerical check of shapes, tie handling, distributed global-batch ranking, bounded weights, and the zero-strength fallback.
5. Do not commit model weights, datasets, generated videos, credentials, machine-specific paths, or benchmark outputs with private paths.
6. Preserve upstream copyright and third-party license headers in modified files.

For scientific claims, provide the exact checkpoint, dataset split, scorer revision, generation settings, seeds, and raw metric output. Report negative results and limitations alongside improvements.
