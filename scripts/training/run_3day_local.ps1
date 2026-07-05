param([double]$BudgetHours = 70, [int]$StaleMinutes = 45)
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

function Left {
    return [math]::Round($BudgetHours - ((Get-Date) - $start).TotalHours, 1)
}

New-Item -ItemType Directory -Force outputs | Out-Null
Write-Output "=== 3-day local chain start $(Get-Date -Format u)  budget ${BudgetHours}h ==="

function Invoke-Guarded([string]$tag, [string]$argList, [string]$logFile, [double]$maxHours, [int]$staleMin = $StaleMinutes) {
    if ($maxHours -le 0.5) {
        Write-Output "SKIP $tag (budget exhausted, ${maxHours}h left)"
        return 0
    }
    $before = Get-Date
    $p = Start-Process -FilePath $py -ArgumentList "-u -m $argList" -PassThru -NoNewWindow `
        -RedirectStandardOutput $logFile -RedirectStandardError "$logFile.err"
    Start-Sleep -Seconds 45  # let start_run create the timestamped run dir
    $hbDir = Get-ChildItem outputs/runs -Directory -EA SilentlyContinue |
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
        "$((Get-Date).ToString('u'))  $tag KILLED: $r" | Out-File -Append outputs/GUARD_KILL.txt -Encoding utf8
        Write-Output "GUARD KILLED ${tag}: $r  (best-by-val checkpoint is safe; --resume to continue)"
        return -1
    }
    Write-Output "$tag exited on its own (code $($p.ExitCode)) at $(Hrs)h"
    return $p.ExitCode
}

function EpochsLogged([string]$pat) {
    $d = Get-ChildItem outputs/runs -Directory -EA SilentlyContinue |
        Where-Object { $_.Name -match $pat } |
        Sort-Object LastWriteTime -Desc |
        Select-Object -First 1
    if (-not $d) {
        return 0
    }
    $m = Join-Path $d.FullName "metrics.jsonl"
    if (-not (Test-Path $m)) {
        return 0
    }
    return (Get-Content $m | Select-String '"epoch"' | Measure-Object).Count
}

if (Test-Path checkpoints/tokenizer/tokenizer_isovocab.pt) {
    Write-Output "`n=== PHASE 1  [$(Hrs)h] SKIP iso-vocab (already done; tokenizer_isovocab.pt exists) ==="
} else {
    Write-Output "`n=== PHASE 1  [$(Hrs)h] iso-vocab tokenizer ==="
    & $py -u -m text2motion.train.train_tokenizer --config configs/tokenizer/tokenizer_isovocab.yaml `
        --tokenizer fsq --ckpt_name tokenizer_isovocab.pt --epochs 50 --eval_every 5 *>&1 |
        Tee-Object outputs/phase1_isovocab.log
}

if (Test-Path checkpoints/generator/generator_transformer_bs8.pt) {
    Write-Output "`n=== PHASE 2  [$(Hrs)h] SKIP transformer-bs8 (done; generator_transformer_bs8.pt exists) ==="
} else {
    Write-Output "`n=== PHASE 2  [$(Hrs)h] transformer-34M bs8 (matched twin partner) ==="
    & $py -u -m text2motion.train.train_generator --config configs/generator/gen_pilot_fsq8x1024.yaml `
        --backbone transformer --tokenizer_ckpt checkpoints/tokenizer/fsq_g8_v1024.pt `
        --init_ckpt checkpoints/generator/generator_transformer_pretrained.pt `
        --ckpt_name generator_transformer_bs8.pt --epochs 18 --batch_size 8 `
        --eval_every 6 --cfg_scale 6.0 --temperature 1.0 --resume *>&1 |
        Tee-Object outputs/phase2_transformer_bs8.log
}

$P3_DONE = "checkpoints/generator/generator_mamba_pretrained.pt.done"
for ($try = 1; $try -le 6; $try++) {
    if (Test-Path $P3_DONE) {
        Write-Output "PHASE 3 complete (marker present)"
        break
    }
    if ((Left) -le 1) {
        Write-Output "PHASE 3 stop: budget exhausted"
        break
    }
    Write-Output "`n=== PHASE 3  [$(Hrs)h, $(Left)h left] mamba pretrain (try $try) ==="
    $maxHours = [math]::Min(14, (Left))
    $code = Invoke-Guarded "phase3_mamba_pretrain" `
        "text2motion.train.train_pretrain --config configs/generator/gen_pilot_fsq8x1024.yaml --backbone mamba --token_pack data/amass_tokens_fsq8x1024.npz --epochs 20 --batch_size 8 --out checkpoints/generator/generator_mamba_pretrained.pt --resume" `
        "outputs/phase3_mamba_pretrain.log" $maxHours
    if ($code -eq 0 -and (Test-Path $P3_DONE)) {
        Write-Output "PHASE 3 complete (clean exit)"
        break
    }
}

if (-not (Test-Path $P3_DONE)) {
    Write-Output "`n=== PHASE 4 SKIP: mamba prior incomplete (no .done marker) ==="
} else {
    for ($try = 1; $try -le 4; $try++) {
        if ((Left) -le 1) {
            Write-Output "PHASE 4 stop: budget exhausted"
            break
        }
        Write-Output "`n=== PHASE 4  [$(Hrs)h, $(Left)h left] mamba bs8 finetune (try $try) ==="
        $code = Invoke-Guarded "phase4_mamba_finetune" `
            "text2motion.train.train_generator --config configs/generator/gen_pilot_fsq8x1024.yaml --backbone mamba --tokenizer_ckpt checkpoints/tokenizer/fsq_g8_v1024.pt --init_ckpt checkpoints/generator/generator_mamba_pretrained.pt --ckpt_name generator_mamba_bs8.pt --epochs 18 --batch_size 4 --grad_accum 2 --eval_every 6 --cfg_scale 6.0 --temperature 1.0 --resume" `
            "outputs/phase4_mamba_finetune.log" (Left)
        if ($code -eq 0) {
            Write-Output "PHASE 4 complete (clean exit)"
            break
        }
    }
}

Write-Output "`n=== CHAIN DONE  total $(Hrs)h ==="
"done $(Get-Date -Format u)  total $(Hrs)h" | Out-File outputs/CHAIN_DONE.txt -Encoding utf8
