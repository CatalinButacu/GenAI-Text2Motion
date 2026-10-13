param([double]$BudgetHours = 8, [int]$StaleMinutes = 45)
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
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"  # anti-frag for the 4GB card
$py = ".\.venv\Scripts\python.exe"
$start = Get-Date

function Hrs {
    return [math]::Round(((Get-Date) - $start).TotalHours, 1)
}

New-Item -ItemType Directory -Force artifacts/outputs | Out-Null
Write-Output "=== transformer+RVQ downstream-generation ablation start $(Get-Date -Format u)  budget ${BudgetHours}h ==="

function Invoke-Guarded([string]$tag, [string]$argList, [string]$logFile, [double]$maxHours, [int]$staleMin = $StaleMinutes) {
    if ($maxHours -le 0.5) {
        Write-Output "SKIP $tag (budget exhausted, ${maxHours}h left)"
        return 0
    }
    $before = Get-Date
    $p = Start-Process -FilePath $py -ArgumentList "-u -m $argList" -PassThru -NoNewWindow `
        -RedirectStandardOutput $logFile -RedirectStandardError "$logFile.err"
    Start-Sleep -Seconds 45  # let start_run create the timestamped run dir
    $hbDir = Get-ChildItem artifacts/outputs/runs -Directory -EA SilentlyContinue |
        Where-Object { $_.LastWriteTime -ge $before } |
        Sort-Object LastWriteTime -Desc |
        Select-Object -First 1
    while (-not $p.HasExited) {
        Start-Sleep -Seconds 120
        $el = ((Get-Date) - $before).TotalHours
        if ($el -ge $maxHours) {
            $r = "max-hours ($maxHours h)"
            break
        }
        if ($hbDir) {
            $newest = Get-ChildItem $hbDir.FullName -File -EA SilentlyContinue |
                Sort-Object LastWriteTime -Descending |
                Select-Object -First 1
            if ($newest) {
                $age = ((Get-Date) - $newest.LastWriteTime).TotalMinutes
                if ($age -ge $staleMin) {
                    $r = "stall (${age}min stale heartbeat)"
                    break
                }
            }
        }
    }
    if (-not $p.HasExited) {
        Stop-Process -Id $p.Id -Force
        Start-Sleep 5
        "$((Get-Date).ToString('u'))  $tag KILLED: $r" | Out-File -Append artifacts/outputs/GUARD_KILL.txt -Encoding utf8
        Write-Output "GUARD KILLED ${tag}: $r  (best-by-val checkpoint is safe; --resume to continue)"
        return -1
    }
    Write-Output "$tag exited on its own (code $($p.ExitCode)) at $(Hrs)h"
    return $p.ExitCode
}

$CONFIG = "configs/generator/gen_pilot_rvq8x1024.yaml"
$TOKENIZER = "artifacts/checkpoints/tokenizer/rvq_l8_1024.pt"
$TOKEN_PACK = "data/amass_tokens_rvq8x1024.npz"
$PRETRAIN_CKPT = "artifacts/checkpoints/generator/generator_transformer_rvq_pretrained.pt"
$GATE = "artifacts/outputs/gates/transformer_rvq_34m_overfit.json"

if (Test-Path artifacts/checkpoints/generator/generator_transformer_rvq_bs8.pt) {
    Write-Output "SKIP: generator_transformer_rvq_bs8.pt already exists"
    exit 0
}

Write-Output "`n=== STAGE 1  [$(Hrs)h] AMASS pretrain, transformer, RVQ tokens (mirrors the FSQ pretrain: 20 epochs, batch 64) ==="
if (-not (Test-Path $PRETRAIN_CKPT)) {
    $code = Invoke-Guarded "pretrain_transformer_rvq" `
        "text2motion.app.cli pretrain --config $CONFIG --backbone transformer --token_pack $TOKEN_PACK --epochs 20 --batch_size 64 --ckpt $PRETRAIN_CKPT --resume" `
        "artifacts/outputs/rvq_ablation_pretrain.log" $BudgetHours
    if ($code -ne 0) {
        Write-Output "PRETRAIN did not finish cleanly -- inspect artifacts/outputs/rvq_ablation_pretrain.log, rerun this script to resume"
        exit 1
    }
} else {
    Write-Output "SKIP pretrain (checkpoint already exists)"
}

Write-Output "`n=== STAGE 2  [$(Hrs)h] sanity-overfit gate (mandatory pre-flight) ==="
if (-not (Test-Path $GATE)) {
    & $py -u -m text2motion.app.cli sanity-overfit --config $CONFIG `
        --backbone transformer --tokenizer_ckpt $TOKENIZER --overfit_gate $GATE
    if ($LASTEXITCODE -ne 0) {
        Write-Output "GATE FAILED -- do not proceed to the real run. Inspect $GATE."
        exit 1
    }
} else {
    Write-Output "SKIP gate (already exists)"
}

Write-Output "`n=== STAGE 3  [$(Hrs)h] HumanML3D fine-tune, transformer, RVQ tokens (mirrors the FSQ twin: 18 epochs, batch 8) ==="
$remaining = $BudgetHours - ((Get-Date) - $start).TotalHours
$code = Invoke-Guarded "train_transformer_rvq" `
    "text2motion.app.cli train-generator --config $CONFIG --backbone transformer --tokenizer_ckpt $TOKENIZER --overfit_gate $GATE --init_ckpt $PRETRAIN_CKPT --ckpt_name generator_transformer_rvq_bs8.pt --epochs 18 --batch_size 8 --num_workers 4 --eval_every 6 --cfg_scale 6.0 --temperature 1.0 --resume" `
    "artifacts/outputs/rvq_ablation_finetune.log" $remaining
if ($code -eq 0) {
    Write-Output "`n=== ABLATION COMPLETE  total $(Hrs)h ==="
    "done $(Get-Date -Format u)  total $(Hrs)h" | Out-File artifacts/outputs/RVQ_ABLATION_DONE.txt -Encoding utf8
} else {
    Write-Output "`n=== FINE-TUNE did not finish cleanly -- inspect artifacts/outputs/rvq_ablation_finetune.log, rerun this script to resume ==="
    exit 1
}
