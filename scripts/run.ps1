# Runs the whole retrieval ablation on this computer's CPU. No API key, no cost.
# Safe to run again: finished steps are skipped.
#
#   powershell -ExecutionPolicy Bypass -File scripts\run.ps1
#
# First run downloads ~1 GB once (PyTorch, two small models, two BEIR datasets), then takes
# roughly 20 to 60 minutes depending on the CPU. Leave the laptop plugged in and awake.
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

function Run($exe, [string[]]$argv) {
    & $exe @argv
    if ($LASTEXITCODE -ne 0) { throw "Failed: $exe $($argv -join ' ')" }
}

if (-not (Get-Command python -ErrorAction SilentlyContinue)) { throw "Python not found. Install Python 3.11 or newer from python.org (tick 'Add to PATH')." }
$ver = & python -c "import sys; print('%d.%d' % sys.version_info[:2])"
if ([version]$ver -lt [version]"3.10") { throw "Python $ver found; 3.10 or newer is needed." }

if (-not (Test-Path ".venv")) { Write-Host "Creating the virtual environment..."; Run python @("-m", "venv", ".venv") }
$py = ".\.venv\Scripts\python.exe"
if (-not (Test-Path ".venv\.installed")) {
    Write-Host "Installing dependencies (PyTorch is large, this takes a while)..."
    Run $py @("-m", "pip", "install", "--quiet", "--upgrade", "pip")
    Run $py @("-m", "pip", "install", "--quiet", "torch", "--index-url", "https://download.pytorch.org/whl/cpu")
    Run $py @("-m", "pip", "install", "--quiet", "-e", ".[dev]")
    Write-Host "Running the tests..."
    Run $py @("-m", "pytest", "-q")
    New-Item ".venv\.installed" -ItemType File | Out-Null
}

$env:TOKENIZERS_PARALLELISM = "false"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
Write-Host "Running the experiments (progress is printed per step)..."
Run $py @("-m", "ablation.cli", "run")
Write-Host "`nDone. Send results\results.json and results\REPORT.md."
