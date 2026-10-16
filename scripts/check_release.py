#!/usr/bin/env python3
"""Check a source snapshot before packaging or publishing."""

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_SUFFIXES = {".mp4", ".mov", ".avi", ".gif", ".safetensors", ".ckpt", ".pt", ".pth", ".bin"}
SOURCE_SUFFIXES = {".py", ".sh", ".ps1", ".yaml", ".yml", ".json", ".md", ".toml", ".txt", ".cff"}
OLD_LABELS = ("la" + "mo", "vm" + "fc")
BRANDING = re.compile("(?i)(?:" + "|".join(
    rf"\b{label}\b|{label}_|_{label}" for label in OLD_LABELS
) + ")")
PRIVATE_PATH = re.compile(r"(?i)(?:/mnt/yrfs/[^/\s]+|/home/[^/\s]+|10\.0\.50\.\d+|hf_[A-Za-z0-9]{20,})")


def main() -> int:
    errors = []
    files = [path for path in ROOT.rglob("*") if path.is_file() and ".git" not in path.parts]
    for path in files:
        rel = path.relative_to(ROOT).as_posix()
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            errors.append(f"large/media artifact: {rel}")
        if path.stat().st_size > 5_000_000:
            errors.append(f"file exceeds 5 MB source budget: {rel}")
        if rel == "scripts/check_release.py":
            continue  # This file intentionally contains the detection patterns.
        if path.suffix.lower() not in SOURCE_SUFFIXES:
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if PRIVATE_PATH.search(content):
            errors.append(f"private path, node address, or token-like value: {rel}")
        if (BRANDING.search(content) and not rel.startswith("eval/benchmark_data/")
                and not rel.startswith("eval/vbench/prompts/")):
            errors.append(f"obsolete project label: {rel}")

    required = ["README.md", "LICENSE", "NOTICE", "CITATION.cff", "requirements.txt",
                "docs/method.md", "docs/reproducibility.md", "scripts/launch_train.py"]
    for name in required:
        if not (ROOT / name).is_file():
            errors.append(f"missing required file: {name}")
    if errors:
        print("Release check failed:\n" + "\n".join(f"- {error}" for error in errors))
        return 1
    print(f"Release check passed: {len(files)} source files; no weights, videos, private paths, or obsolete labels")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
