# Head-to-head TEST evaluation of the two 34M pilots (transformer bs8 vs mamba bs8), on the frozen
# FSQ 8x1024 tokenizer, under the identical citable 20-rep protocol. Waits for the phase-4 finetune
# to release the 4GB GPU (only one job fits), then scores both, then MultiModality.
#
# Locked, identical for BOTH twins (the whole point is a fair comparison):
#   split test (4384 clips) | cfg_scale 6.0 | temperature 1.0 | length_mode fixed (GT length,
#   field-standard) | tokenizer fsq_g8_v1024 | config gen_pilot_fsq8x1024 | seed from config.
#
#   powershell -File scripts/eval/twin_test_eval.ps1            # waits then runs
#   powershell -File scripts/eval/twin_test_eval.ps1 -NoWait    # GPU already free
param([switch]$NoWait, [double]$MaxWaitHours = 10)
$ErrorActionPreference = "Continue"
$root = $PSScriptRoot
while ($root -and -not (Test-Path (Join-Path $root "pyproject.toml"))) { $root = Split-Path $root -Parent }
if (-not $root) { throw "repo root (pyproject.toml) not found above $PSScriptRoot" }
Set-Location $root
$env:PYTHONPATH = "src"
$env:PYTORCH_CUDA_ALLOC_CONF = "max_split_size_mb:64"
$py = ".\.venv\Scripts\python.exe"

$config = "configs/generator/gen_pilot_fsq8x1024.yaml"
$tok = "checkpoints/tokenizer/fsq_g8_v1024.pt"
$transformerCkpt = "checkpoints/generator/generator_transformer_bs8.pt"
$mambaCkpt = "checkpoints/generator_mamba_bs8.pt"

function Gpu-FreeMiB {
    $used = (nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) | Select-Object -First 1
    return [int]$used
}

if (-not $NoWait) {
    # the finetune sits at ~3GB; wait until it exits and the card falls back to the desktop baseline
    Write-Output "waiting for the 4GB GPU to free (finetune to finish)... $(Get-Date -Format u)"
    $start = Get-Date
    while ($true) {
        Start-Sleep -Seconds 120
        $mib = Gpu-FreeMiB
        $chainDone = (Test-Path outputs/CHAIN_DONE.txt) -and
            ((Get-Item outputs/CHAIN_DONE.txt).LastWriteTime -ge $start)
        if ($mib -lt 1500 -and (Test-Path $mambaCkpt)) {
            Write-Output "GPU free (${mib} MiB) + mamba ckpt present -> starting eval $(Get-Date -Format u)"
            break
        }
        if ($chainDone) { Write-Output "CHAIN_DONE written; GPU ${mib} MiB"; }
        if (((Get-Date) - $start).TotalHours -ge $MaxWaitHours) {
            Write-Output "max wait ($MaxWaitHours h) reached; proceeding anyway (GPU ${mib} MiB)"
            break
        }
    }
    Start-Sleep -Seconds 20  # let the training process fully release CUDA
}

function Eval-One([string]$backbone, [string]$ckpt) {
    if (-not (Test-Path $ckpt)) { Write-Output "SKIP $backbone : missing $ckpt"; return }
    Write-Output "`n=== TEST eval: $backbone  ($ckpt)  $(Get-Date -Format u) ==="
    & $py -u -m text2motion.eval.evaluate `
        --config $config --backbone $backbone --ckpt $ckpt --tokenizer_ckpt $tok `
        --split test --cfg_scale 6.0 --temperature 1.0 --length_mode fixed `
        --mm_clips 100 --mm_repeats 30 *>&1 | Tee-Object "outputs/twin_test_$backbone.log"
}

Eval-One "transformer" $transformerCkpt
Eval-One "mamba" $mambaCkpt
Write-Output "`n=== TWIN TEST EVAL DONE  $(Get-Date -Format u) ==="
"done $(Get-Date -Format u)" | Out-File outputs/TWIN_TEST_DONE.txt -Encoding utf8
