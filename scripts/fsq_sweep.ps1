# Seeded FSQ sweep — runs AFTER the RVQ sweep finishes (waits on its sentinel). Fresh seeded runs
# (seed 2026, new fsq_* ckpt names so it does NOT resume the old unseeded checkpoints). Mirrors the
# RVQ sweep so both tokenizers are reproducible at seed 2026. Core configs first (winner + iso-vocab),
# then the latent-space sweep. 500 ep, eval 25, resume-safe, manifest-logged.
# Continue (not Stop): python writes warnings to stderr, and `2>&1 | Out-File` wraps each stderr line
# as an ErrorRecord -> with Stop that terminates the driver. Continue keeps it running.
$ErrorActionPreference = "Continue"
$env:PYTHONPATH = "src"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"  # guard vs fragmentation OOM (4GB GPU)
$py = ".venv\Scripts\python.exe"

# 1) wait for the RVQ sweep to complete (frees the GPU)
while (-not (Test-Path "outputs/rvq_sweep_done.txt")) { Start-Sleep -Seconds 600 }
"=== $(Get-Date -Format o)  RVQ done -> starting seeded FSQ sweep ===" | Out-File -Append outputs/fsq_sweep_start.txt -Encoding utf8

$variants = @(
    @{ cfg = "configs/tok_g6_v1000.yaml";     name = "fsq_g6_v1000.pt" },   # winner 6x1000
    @{ cfg = "configs/tokenizer_isovocab.yaml"; name = "fsq_g6_v512.pt" },  # iso-vocab 6x512 (H1a)
    @{ cfg = "configs/tok_g4_v1000.yaml";      name = "fsq_g4_v1000.pt" },  # latent sweep
    @{ cfg = "configs/tok_g8_v1000.yaml";      name = "fsq_g8_v1000.pt" },
    @{ cfg = "configs/tok_g6_v2560.yaml";      name = "fsq_g6_v2560.pt" }
)

foreach ($v in $variants) {
    $stem = [System.IO.Path]::GetFileNameWithoutExtension($v.name)
    $log = "outputs/sweep_$stem.log"
    "=== $(Get-Date -Format o)  training $stem ===" | Out-File -Append $log -Encoding utf8
    & $py -u -m text2motion.train.train_tokenizer --config $v.cfg --tokenizer fsq `
        --epochs 500 --batch_size 128 --eval_every 25 --ckpt_name $v.name --resume `
        2>&1 | Out-File -Append $log -Encoding utf8
    "=== $(Get-Date -Format o)  done $stem ===" | Out-File -Append $log -Encoding utf8
}
"=== FSQ SWEEP COMPLETE ===" | Out-File outputs/fsq_sweep_done.txt -Encoding utf8
