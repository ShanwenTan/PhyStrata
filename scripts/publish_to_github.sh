#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo 'Usage: bash scripts/publish_to_github.sh OWNER/PhyStrata' >&2
  exit 2
fi
cd "$(dirname "$0")/.."
command -v git >/dev/null || { echo 'Install Git first.' >&2; exit 1; }
command -v gh >/dev/null || { echo 'Install GitHub CLI and run gh auth login first.' >&2; exit 1; }
gh auth status >/dev/null
python scripts/check_release.py
if [[ ! -d .git ]]; then git init -b main; fi
git add -A
git commit -m 'Initial PhyStrata research source release'
gh repo create "$1" --public --source . --remote origin --push
