param([switch]$NoWait, [double]$MaxWaitHours = 10)
$ErrorActionPreference = "Continue"
$root = $PSScriptRoot
while ($root -and -not (Test-Path (Join-Path $root "pyproject.toml"))) {
    $root = Split-Path $root -Parent
}
if (-not $root) {
    throw "repo root (pyproject.toml) not found above $PSScriptRoot"
}
Set-Location $root
$env:PYTHONPATH = "src"
$env:PYTORCH_CUDA_ALLOC_CONF = "max_split_size_mb:64"
$py = ".\.venv\Scripts\python.exe"

$config = "configs/generator/gen_pilot_fsq8x1024.yaml"
$tok = "checkpoints/tokenizer/fsq_g8_v1024.pt"
$transformerCkpt = "checkpoints/generator/generator_transformer_bs8.pt"
$mambaCkpt = "checkpoints/generator_mamba_bs8.pt"

function Get-GpuFreeMiB {
    $used = (nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) | Select-Object -First 1
    return [int]$used
}

if (-not $NoWait) {
    Write-Output "waiting for the 4GB GPU to free (finetune to finish)... $(Get-Date -Format u)"
    $start = Get-Date
    while ($true) {
        Start-Sleep -Seconds 120
        $mib = Get-GpuFreeMiB
        $chainFresh = (Test-Path outputs/CHAIN_DONE.txt) -and ((Get-Item outputs/CHAIN_DONE.txt).LastWriteTime -ge $start)
        if ($mib -lt 1500 -and (Test-Path $mambaCkpt)) {
            Write-Output "GPU free (${mib} MiB) + mamba ckpt present -> starting eval $(Get-Date -Format u)"
            break
        }
        if ($chainFresh) {
            Write-Output "CHAIN_DONE written; GPU ${mib} MiB"
        }
        if (((Get-Date) - $start).TotalHours -ge $MaxWaitHours) {
            Write-Output "max wait ($MaxWaitHours h) reached; proceeding anyway (GPU ${mib} MiB)"
            break
        }
    }
    Start-Sleep -Seconds 20  # let the training process fully release CUDA
}

function Invoke-Eval([string]$backbone, [string]$ckpt) {
    if (-not (Test-Path $ckpt)) {
        Write-Output "SKIP $backbone : missing $ckpt"
        return
    }
    Write-Output "`n=== TEST eval: $backbone  ($ckpt)  $(Get-Date -Format u) ==="
    & $py -u -m text2motion.app.cli evaluate `
        --config $config --backbone $backbone --ckpt $ckpt --tokenizer_ckpt $tok `
        --split test --cfg_scale 6.0 --temperature 1.0 --length_mode fixed `
        --mm_clips 100 --mm_repeats 30 *>&1 | Tee-Object "outputs/twin_test_$backbone.log"
}

Invoke-Eval "transformer" $transformerCkpt
Invoke-Eval "mamba" $mambaCkpt
Write-Output "`n=== TWIN TEST EVAL DONE  $(Get-Date -Format u) ==="
"done $(Get-Date -Format u)" | Out-File outputs/TWIN_TEST_DONE.txt -Encoding utf8
