param([double]$BudgetHoursPerRun = 72, [int]$StaleMinutes = 90)
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
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
$py = ".\.venv\Scripts\python.exe"
$start = Get-Date

function Hrs {
    return [math]::Round(((Get-Date) - $start).TotalHours, 1)
}

New-Item -ItemType Directory -Force artifacts/outputs | Out-Null
Write-Output "=== matched 60-epoch twin extension start $(Get-Date -Format u) ==="

function Invoke-Guarded([string]$tag, [string]$argList, [string]$logFile, [double]$maxHours, [int]$staleMin = $StaleMinutes) {
    if ($maxHours -le 0.5) {
        Write-Output "SKIP $tag (budget exhausted, ${maxHours}h left)"
        return 0
    }
    $before = Get-Date
    $p = Start-Process -FilePath $py -ArgumentList "-u -m $argList" -PassThru -NoNewWindow `
        -RedirectStandardOutput $logFile -RedirectStandardError "$logFile.err"
    Start-Sleep -Seconds 45
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
    Write-Output "$tag exited on its own (code $($p.ExitCode)) at $(Hrs)h -- verify via checkpoint, exit code is unreliable on this machine"
    return 0
}

function Run-Twin([string]$label, [string]$config, [string]$tokenizer, [string]$initCkpt, [string]$gate, [string]$ckptName) {
    $target = "artifacts/checkpoints/generator/$ckptName"
    if (Test-Path $target) {
        Write-Output "SKIP $label (checkpoint already exists: $ckptName)"
        return
    }
    Write-Output "`n=== $label  [$(Hrs)h] 60-epoch fine-tune (transformer, batch 8, matched recipe) ==="
    Invoke-Guarded $label `
        "text2motion.app.cli train-generator --config $config --backbone transformer --tokenizer_ckpt $tokenizer --overfit_gate $gate --init_ckpt $initCkpt --ckpt_name $ckptName --epochs 60 --batch_size 8 --num_workers 4 --eval_every 6 --cfg_scale 6.0 --temperature 1.0 --resume" `
        "artifacts/outputs/$label.log" $BudgetHoursPerRun | Out-Null
}

Run-Twin "twin60_transformer_fsq" `
    "configs/generator/gen_pilot_fsq8x1024.yaml" `
    "artifacts/checkpoints/tokenizer/fsq_g8_v1024.pt" `
    "artifacts/checkpoints/generator/generator_transformer_pretrained.pt" `
    "artifacts/outputs/gates/transformer_34m_overfit.json" `
    "generator_transformer_bs8_e60.pt"

Run-Twin "twin60_transformer_rvq" `
    "configs/generator/gen_pilot_rvq8x1024.yaml" `
    "artifacts/checkpoints/tokenizer/rvq_l8_1024.pt" `
    "artifacts/checkpoints/generator/generator_transformer_rvq_pretrained.pt" `
    "artifacts/outputs/gates/transformer_rvq_34m_overfit.json" `
    "generator_transformer_rvq_bs8_e60.pt"

function Run-Mamba60([string]$label, [string]$ckptName) {
    $target = "artifacts/checkpoints/generator/$ckptName"
    if (Test-Path $target) {
        Write-Output "SKIP $label (checkpoint already exists: $ckptName)"
        return
    }
    Write-Output "`n=== $label  [$(Hrs)h] 60-epoch fine-tune (mamba, batch 8, eager scan -- no fused kernel on Windows) ==="
    Invoke-Guarded $label `
        "text2motion.app.cli train-generator --config configs/generator/gen_pilot_fsq8x1024.yaml --backbone mamba --tokenizer_ckpt artifacts/checkpoints/tokenizer/fsq_g8_v1024.pt --overfit_gate artifacts/outputs/gates/mamba_34m_overfit.json --init_ckpt artifacts/checkpoints/generator/generator_mamba_pretrained.pt --ckpt_name $ckptName --epochs 60 --batch_size 8 --num_workers 4 --eval_every 6 --cfg_scale 6.0 --temperature 1.0 --resume" `
        "artifacts/outputs/$label.log" $BudgetHoursPerRun | Out-Null
}

Run-Mamba60 "twin60_mamba_fsq" "generator_mamba_bs8_e60.pt"

Write-Output "`n=== ALL STAGES ATTEMPTED  total $(Hrs)h -- verify checkpoints directly, do not trust this script's own exit code ==="
"done $(Get-Date -Format u)  total $(Hrs)h" | Out-File artifacts/outputs/MATCHED_TWIN_60EP_DONE.txt -Encoding utf8
