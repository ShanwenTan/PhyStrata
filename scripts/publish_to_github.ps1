param(
    [Parameter(Mandatory = $true)]
    [string]$Repository
)

$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location -LiteralPath $root

if (-not (Get-Command git -ErrorAction SilentlyContinue)) { throw 'Install Git first.' }
if (-not (Get-Command gh -ErrorAction SilentlyContinue)) { throw 'Install GitHub CLI and run gh auth login first.' }
gh auth status *> $null
if ($LASTEXITCODE -ne 0) { throw 'Run gh auth login first.' }

python scripts/check_release.py
if ($LASTEXITCODE -ne 0) { throw 'Release validation failed.' }

if (-not (Test-Path -LiteralPath '.git')) { git init -b main }
git add -A
git commit -m 'Initial PhyStrata research source release'
if ($LASTEXITCODE -ne 0) { throw 'Git commit failed; configure your Git identity or inspect the repository.' }
gh repo create $Repository --public --source . --remote origin --push
if ($LASTEXITCODE -ne 0) { throw 'GitHub repository creation or push failed.' }
