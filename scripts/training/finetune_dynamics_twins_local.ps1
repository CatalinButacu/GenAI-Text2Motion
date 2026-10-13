param(
    [double]$BudgetHoursPerRun = 48,
    [int]$StaleMinutes = 90,
    [int]$Epochs = 72,
    [string]$SeedFrom = "e60",
    [switch]$ContinueSelection
)
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
$config = "configs/generator/gen_pilot_fsq8x1024_dynamics.yaml"
$tokenizer = "artifacts/checkpoints/tokenizer/fsq_g8_v1024.pt"
$checkpoints = "artifacts/checkpoints/generator"

function Hrs {
    return [math]::Round(((Get-Date) - $start).TotalHours, 1)
}

New-Item -ItemType Directory -Force artifacts/outputs | Out-Null
Write-Output "=== dynamics fine-tune, both twins, from $SeedFrom -> epoch $Epochs, continue selection $ContinueSelection, start $(Get-Date -Format u) ==="

function Invoke-Guarded([string]$tag, [string]$argList, [string]$logFile, [double]$maxHours) {
    $before = Get-Date
    $p = Start-Process -FilePath $py -ArgumentList "-u -m $argList" -PassThru -NoNewWindow `
        -RedirectStandardOutput $logFile -RedirectStandardError "$logFile.err"
    Start-Sleep -Seconds 45
    $hbDir = Get-ChildItem artifacts/logs/train -Directory -EA SilentlyContinue |
        Where-Object { $_.LastWriteTime -ge $before } |
        Sort-Object LastWriteTime -Desc |
        Select-Object -First 1
    $reason = ""
    while (-not $p.HasExited) {
        Start-Sleep -Seconds 120
        if (((Get-Date) - $before).TotalHours -ge $maxHours) {
            $reason = "max-hours ($maxHours h)"
            break
        }
        if ($hbDir) {
            $newest = Get-ChildItem $hbDir.FullName -File -EA SilentlyContinue |
                Sort-Object LastWriteTime -Descending |
                Select-Object -First 1
            if ($newest) {
                $age = ((Get-Date) - $newest.LastWriteTime).TotalMinutes
                if ($age -ge $StaleMinutes) {
                    $reason = "stall (${age}min stale heartbeat)"
                    break
                }
            }
        }
    }
    if (-not $p.HasExited) {
        Stop-Process -Id $p.Id -Force
        Start-Sleep 5
        "$((Get-Date).ToString('u'))  $tag KILLED: $reason" | Out-File -Append artifacts/outputs/GUARD_KILL.txt -Encoding utf8
        Write-Output "GUARD KILLED ${tag}: $reason (rerun this script to resume)"
        return -1
    }
    Write-Output "$tag exited on its own at $(Hrs)h -- verify via checkpoint, exit code is unreliable on this machine"
    return 0
}

function Run-FineTune([string]$backbone, [int]$numWorkers) {
    $label = "finetune_dynamics_${backbone}_e$Epochs"
    $donePath = "artifacts/outputs/${label}_DONE.txt"
    if (Test-Path $donePath) {
        Write-Output "SKIP $label (done marker present: $donePath)"
        return
    }
    $name = "generator_${backbone}_bs8_dyn_e$Epochs.pt"
    $resume = Join-Path $checkpoints ([IO.Path]::GetFileNameWithoutExtension($name) + "_last.pt")
    if (-not (Test-Path $resume)) {
        $seed = Join-Path $checkpoints "generator_${backbone}_bs8_${SeedFrom}_last.pt"
        if (-not (Test-Path $seed)) {
            throw "seed state $seed not found"
        }
        Copy-Item $seed $resume
        Write-Output "seeded $resume from $seed"
    }
    $gate = "artifacts/outputs/gates/${backbone}_34m_dynamics_overfit.json"
    $selection = if ($ContinueSelection) { "" } else { "--fresh_selection" }
    Write-Output "`n=== $label  [$(Hrs)h] ($backbone, batch 8, num_workers $numWorkers) ==="
    $code = @(Invoke-Guarded $label `
        "text2motion.app.cli train-generator --config $config --backbone $backbone --tokenizer_ckpt $tokenizer --overfit_gate $gate --ckpt_name $name --epochs $Epochs --batch_size 8 --num_workers $numWorkers --eval_every 6 --cfg_scale 6.0 --temperature 1.0 --resume $selection" `
        "artifacts/outputs/$label.log" $BudgetHoursPerRun)[-1]
    $reached = & $py -c "import torch; print(torch.load(r'$resume', map_location='cpu', weights_only=False)['epoch'] + 1)"
    if ($code -eq 0 -and [int]$reached -ge $Epochs) {
        "done $(Get-Date -Format u) at epoch $reached" | Out-File $donePath -Encoding utf8
        Write-Output "$label complete: epoch $reached of $Epochs"
    } else {
        Write-Output "$label INCOMPLETE: checkpoint at epoch $reached of $Epochs -- see artifacts/outputs/$label.log.err; rerun to resume"
    }
}

Run-FineTune "transformer" 4
Run-FineTune "mamba" 2

Write-Output "`n=== BOTH TWINS ATTEMPTED  total $(Hrs)h -- verify checkpoints directly ==="
"done $(Get-Date -Format u)  total $(Hrs)h" | Out-File artifacts/outputs/FINETUNE_DYNAMICS_DONE.txt -Encoding utf8
