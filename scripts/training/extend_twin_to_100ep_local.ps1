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
Write-Output "=== extend twin 60ep -> 100ep start $(Get-Date -Format u) ==="

function Invoke-Guarded([string]$tag, [string]$argList, [string]$logFile, [double]$maxHours, [int]$staleMin = $StaleMinutes) {
    if ($maxHours -le 0.5) {
        Write-Output "SKIP $tag (budget exhausted, ${maxHours}h left)"
        return 0
    }
    $before = Get-Date
    $p = Start-Process -FilePath $py -ArgumentList "-u -m $argList" -PassThru -NoNewWindow `
        -RedirectStandardOutput $logFile -RedirectStandardError "$logFile.err"
    Start-Sleep -Seconds 45
    $hbDir = Get-ChildItem artifacts/logs/train -Directory -EA SilentlyContinue |
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

function Run-Extend([string]$label, [string]$backbone, [string]$config, [string]$tokenizer, [string]$gate, [string]$ckptName, [int]$numWorkers) {
    $target = "artifacts/checkpoints/generator/$ckptName"
    $donePath = "artifacts/outputs/${label}_DONE.txt"
    if (Test-Path $donePath) {
        Write-Output "SKIP $label (done marker present: $donePath)"
        return
    }
    Write-Output "`n=== $label  [$(Hrs)h] extend to 100 epochs ($backbone, batch 8, num_workers $numWorkers) ==="
    $code = Invoke-Guarded $label `
        "text2motion.app.cli train-generator --config $config --backbone $backbone --tokenizer_ckpt $tokenizer --overfit_gate $gate --ckpt_name $ckptName --epochs 100 --batch_size 8 --num_workers $numWorkers --eval_every 6 --cfg_scale 6.0 --temperature 1.0 --resume" `
        "artifacts/outputs/$label.log" $BudgetHoursPerRun
    if ($code -eq 0) {
        "done $(Get-Date -Format u)" | Out-File $donePath -Encoding utf8
    }
}

Run-Extend "extend100_transformer_fsq" "transformer" `
    "configs/generator/gen_pilot_fsq8x1024.yaml" `
    "artifacts/checkpoints/tokenizer/fsq_g8_v1024.pt" `
    "artifacts/outputs/gates/transformer_34m_overfit.json" `
    "generator_transformer_bs8_e100.pt" `
    4

Run-Extend "extend100_mamba_fsq" "mamba" `
    "configs/generator/gen_pilot_fsq8x1024.yaml" `
    "artifacts/checkpoints/tokenizer/fsq_g8_v1024.pt" `
    "artifacts/outputs/gates/mamba_34m_overfit.json" `
    "generator_mamba_bs8_e100.pt" `
    2

Write-Output "`n=== ALL STAGES ATTEMPTED  total $(Hrs)h -- verify checkpoints directly, do not trust this script's own exit code ==="
"done $(Get-Date -Format u)  total $(Hrs)h" | Out-File artifacts/outputs/EXTEND_TWIN_100EP_DONE.txt -Encoding utf8
