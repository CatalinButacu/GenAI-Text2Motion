# 3-day local training chain (user has the PC free; launch once, runs unattended). Front-loads the
# two GUARANTEED-fast artifacts, then the Mamba prior, then the long Mamba finetune (resumable +
# best-by-val, so a budget cutoff at any point loses nothing). Every stage logs to outputs/phaseN_*.log
# and its own run_log metrics.jsonl. Eager Mamba is the bottleneck: ~3 h/epoch at bs8 on the 4 GB GPU
# (above bs8 the parallel scan spills to shared RAM over PCIe -> 25x slower), so the Mamba twin is a
# BEST-EFFORT, reduced-budget point -- the budget-matched twin is the cloud run. See README.
#
#   ! powershell -ExecutionPolicy Bypass -File scripts\run_3day_local.ps1
#
# Long phases (3,4) run under an inline watchdog: killed if they exceed the remaining wall-clock
# budget OR stall (metrics.jsonl heartbeat goes stale) -- the laptop twin of the cloud cost guards
# (CLAUDE.md). Resume after any interruption: just relaunch -- stages skip or --resume cleanly.
param([double]$BudgetHours = 70, [int]$StaleMinutes = 45)

$ErrorActionPreference = "Continue"
Set-Location (Split-Path $PSScriptRoot -Parent)
$env:PYTHONPATH = "src"
$env:PYTORCH_CUDA_ALLOC_CONF = "max_split_size_mb:64"  # aggressive anti-frag for the 4GB card
$py = ".\.venv\Scripts\python.exe"
$start = Get-Date
function Hrs { [math]::Round(((Get-Date) - $start).TotalHours, 1) }
function Left { [math]::Round($BudgetHours - ((Get-Date) - $start).TotalHours, 1) }
New-Item -ItemType Directory -Force outputs | Out-Null
Write-Output "=== 3-day local chain start $(Get-Date -Format u)  budget ${BudgetHours}h ==="

# Run a python stage under a wall-clock + stall watchdog. $argList is the module + args (one string).
# Stall = the run dir's NEWEST file goes untouched for $staleMin (train_* writes a per-100-step
# heartbeat file, so a slow 3h epoch is NOT mistaken for a hang).
function Invoke-Guarded([string]$tag, [string]$argList, [string]$logFile, [double]$maxHours, [int]$staleMin = $StaleMinutes) {
    if ($maxHours -le 0.5) { Write-Output "SKIP $tag (budget exhausted, ${maxHours}h left)"; return 0 }
    $before = Get-Date
    $p = Start-Process -FilePath $py -ArgumentList "-u -m $argList" -PassThru -NoNewWindow `
        -RedirectStandardOutput $logFile -RedirectStandardError "$logFile.err"
    Start-Sleep -Seconds 45  # let start_run create the timestamped run dir
    $hbDir = Get-ChildItem outputs/runs -Directory -EA SilentlyContinue |
        Where-Object { $_.LastWriteTime -ge $before } | Sort-Object LastWriteTime -Desc |
        Select-Object -First 1
    while (-not $p.HasExited) {
        Start-Sleep -Seconds 120
        $el = ((Get-Date) - $before).TotalHours
        if ($el -ge $maxHours) { $r = "max-hours ($maxHours h)"; break }
        if ($hbDir) {
            $newest = Get-ChildItem $hbDir.FullName -File -EA SilentlyContinue |
                Sort-Object LastWriteTime -Descending | Select-Object -First 1
            if ($newest) {
                $age = ((Get-Date) - $newest.LastWriteTime).TotalMinutes
                if ($age -ge $staleMin) { $r = "stall (${age}min stale heartbeat)"; break }
            }
        }
    }
    if (-not $p.HasExited) {
        Stop-Process -Id $p.Id -Force; Start-Sleep 5
        "$((Get-Date).ToString('u'))  $tag KILLED: $r" | Out-File -Append outputs/GUARD_KILL.txt -Encoding utf8
        Write-Output "GUARD KILLED ${tag}: $r  (best-by-val checkpoint is safe; --resume to continue)"
        return -1
    }
    Write-Output "$tag exited on its own (code $($p.ExitCode)) at $(Hrs)h"
    return $p.ExitCode
}

# Epochs logged so far by the newest run dir matching $pat (crash/resume progress probe).
function EpochsLogged([string]$pat) {
    $d = Get-ChildItem outputs/runs -Directory -EA SilentlyContinue | Where-Object { $_.Name -match $pat } |
        Sort-Object LastWriteTime -Desc | Select-Object -First 1
    if (-not $d) { return 0 }
    $m = Join-Path $d.FullName "metrics.jsonl"
    if (-not (Test-Path $m)) { return 0 }
    (Get-Content $m | Select-String '"epoch"' | Measure-Object).Count
}

# --- PHASE 1: iso-vocab FSQ tokenizer (Contribution A codebook-size confound row) ~0.3h ---
if (Test-Path checkpoints/tokenizer/tokenizer_isovocab.pt) {
    Write-Output "`n=== PHASE 1  [$(Hrs)h] SKIP iso-vocab (already done; tokenizer_isovocab.pt exists) ==="
} else {
    Write-Output "`n=== PHASE 1  [$(Hrs)h] iso-vocab tokenizer ==="
    & $py -u -m text2motion.train.train_tokenizer --config configs/tokenizer/tokenizer_isovocab.yaml `
      --tokenizer fsq --ckpt_name tokenizer_isovocab.pt --epochs 50 --eval_every 5 *>&1 |
      Tee-Object outputs/phase1_isovocab.log
}

# --- PHASE 2: transformer-34M bs8 twin partner (from the AMASS prior we already have) ~1.5h ---
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

# --- PHASE 3: Mamba-34M AMASS pretrain (motion prior). Eager mamba OOMs ~ep15 from fragmentation;
# each --resume starts a FRESH process (clears fragmentation) and continues -> retry until 20 epochs. ---
# --- PHASE 3: Mamba-34M AMASS pretrain (motion prior). Eager mamba OOMs ~ep15 from fragmentation;
# each --resume starts a FRESH process (clears fragmentation) and continues -> retry until the
# completion marker (.done) appears. ---
$P3_DONE = "checkpoints/generator/generator_mamba_pretrained.pt.done"
for ($try = 1; $try -le 6; $try++) {
    if (Test-Path $P3_DONE) { Write-Output "PHASE 3 complete (marker present)"; break }
    if ((Left) -le 1) { Write-Output "PHASE 3 stop: budget exhausted"; break }
    Write-Output "`n=== PHASE 3  [$(Hrs)h, $(Left)h left] mamba pretrain (try $try) ==="
    $code = Invoke-Guarded "phase3_mamba_pretrain" `
      "text2motion.train.train_pretrain --config configs/generator/gen_pilot_fsq8x1024.yaml --backbone mamba --token_pack data/amass_tokens_fsq8x1024.npz --epochs 20 --batch_size 8 --out checkpoints/generator/generator_mamba_pretrained.pt --resume" `
      "outputs/phase3_mamba_pretrain.log" ([math]::Min(14, (Left)))
    if ($code -eq 0 -and (Test-Path $P3_DONE)) { Write-Output "PHASE 3 complete (clean exit)"; break }
}

# --- PHASE 4: Mamba-34M bs8 finetune (the twin; long pole, resumable + best-by-val) ~rest ---
# ~3h/epoch eager; the 100-step heartbeat keeps the default stall guard valid. Retry on real crash
# (--resume continues from the per-epoch _last); exit 0 = all 18 epochs done.
if (-not (Test-Path $P3_DONE)) {
    Write-Output "`n=== PHASE 4 SKIP: mamba prior incomplete (no .done marker) ==="
} else {
    for ($try = 1; $try -le 4; $try++) {
        if ((Left) -le 1) { Write-Output "PHASE 4 stop: budget exhausted"; break }
        Write-Output "`n=== PHASE 4  [$(Hrs)h, $(Left)h left] mamba bs8 finetune (try $try) ==="
        $code = Invoke-Guarded "phase4_mamba_finetune" `
          "text2motion.train.train_generator --config configs/generator/gen_pilot_fsq8x1024.yaml --backbone mamba --tokenizer_ckpt checkpoints/tokenizer/fsq_g8_v1024.pt --init_ckpt checkpoints/generator/generator_mamba_pretrained.pt --ckpt_name generator_mamba_bs8.pt --epochs 18 --batch_size 8 --eval_every 6 --cfg_scale 6.0 --temperature 1.0 --resume" `
          "outputs/phase4_mamba_finetune.log" (Left)
        if ($code -eq 0) { Write-Output "PHASE 4 complete (clean exit)"; break }
    }
}

Write-Output "`n=== CHAIN DONE  total $(Hrs)h ==="
"done $(Get-Date -Format u)  total $(Hrs)h" | Out-File outputs/CHAIN_DONE.txt -Encoding utf8
