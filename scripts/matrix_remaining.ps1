# Completes the tokenizer matrix: the 5 cells still missing/partial after the Jun 16-17 sweep.
# RUN THIS YOURSELF in a normal terminal (NOT as a harness background task -- the duplicate-driver
# collision on Jun 17 OOM'd the 4GB GPU and crashed fsq_g4_v1024 mid-run). Single instance only.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\matrix_remaining.ps1
#
# All runs: seed 2026, strict determinism, 500 ep, eval every 25, bs 128, resume-safe, manifest-logged.
# 3 fresh + 2 resumes (fsq_g4_v1024 ep286->500, fsq_g4_v1000 ep72->500). Sequential -- 4GB GPU fits
# exactly one tokenizer at a time. Est. ~2-3 h per fresh run on the RTX 3050.

$ErrorActionPreference = "Continue"
$env:PYTHONPATH = "src"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
$py = ".venv\Scripts\python.exe"

# Single-instance guard: refuse to start if a tokenizer trainer is already running (prevents the
# 4GB-GPU collision that crashed the last run).
$running = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Where-Object { $_.CommandLine -like '*train_tokenizer*' })
if ($running.Count -gt 0) {
    Write-Host "ABORT: a train_tokenizer process is already running (PID $($running[0].ProcessId)). Stop it first." -ForegroundColor Red
    exit 1
}

# cfg, mechanism, ckpt name. Fresh cells first (complete the 8-row), then the two partial resumes.
$variants = @(
    @{ cfg = "configs/tokenizer/rvq_l8_1024.yaml";  mech = "rvq"; name = "rvq_l8_1024.pt" },   # RVQ 8x1024  fresh
    @{ cfg = "configs/tokenizer/fsq_g8_v1024.yaml"; mech = "fsq"; name = "fsq_g8_v1024.pt" },  # FSQ 8x1024  fresh
    @{ cfg = "configs/tokenizer/tok_g8_v1000.yaml"; mech = "fsq"; name = "fsq_g8_v1000.pt" },  # FSQ 8x1000  fresh
    @{ cfg = "configs/tokenizer/fsq_g4_v1024.yaml"; mech = "fsq"; name = "fsq_g4_v1024.pt" },  # FSQ 4x1024  resume ep286
    @{ cfg = "configs/tokenizer/tok_g4_v1000.yaml"; mech = "fsq"; name = "fsq_g4_v1000.pt" }   # FSQ 4x1000  resume ep72
)

foreach ($v in $variants) {
    $stem = [System.IO.Path]::GetFileNameWithoutExtension($v.name)
    $log  = "outputs/sweep_$stem.log"
    Write-Host "=== $(Get-Date -Format o)  training $stem ($($v.mech)) ===" -ForegroundColor Cyan
    "=== $(Get-Date -Format o)  training $stem ($($v.mech)) ===" | Out-File -Append $log -Encoding utf8
    & $py -u -m text2motion.train.train_tokenizer --config $v.cfg --tokenizer $v.mech `
        --epochs 500 --batch_size 128 --eval_every 25 --ckpt_name $v.name --resume `
        2>&1 | Tee-Object -FilePath $log -Append
    "=== $(Get-Date -Format o)  done $stem ===" | Out-File -Append $log -Encoding utf8
}
"=== MATRIX REMAINING COMPLETE ===" | Out-File outputs/matrix_remaining_done.txt -Encoding utf8
Write-Host "ALL 5 REMAINING RUNS COMPLETE." -ForegroundColor Green
