$ErrorActionPreference = "Continue"
$env:PYTHONPATH = "src"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"  # guard vs fragmentation OOM (4GB GPU)
$py = ".venv\Scripts\python.exe"
while (-not (Test-Path "outputs/rvq_sweep_done.txt")) {
    Start-Sleep -Seconds 600
}
"=== $(Get-Date -Format o)  starting seeded FSQ sweep ===" | Out-File -Append outputs/fsq_sweep_start.txt -Encoding utf8
$variants = @(
    @{ cfg = "configs/tokenizer/tokenizer_isovocab.yaml"; name = "fsq_g6_v512.pt" },   # <-> rvq_l6_512  (6x512)
    @{ cfg = "configs/tokenizer/fsq_g6_v1024.yaml";       name = "fsq_g6_v1024.pt" },  # <-> rvq_l6_1024 (6x1024)
    @{ cfg = "configs/tokenizer/fsq_g4_v512.yaml";        name = "fsq_g4_v512.pt" },   # <-> rvq_l4_512  (4x512)
    @{ cfg = "configs/tokenizer/fsq_g8_v512.yaml";        name = "fsq_g8_v512.pt" },   # <-> rvq_l8_512  (8x512)
    @{ cfg = "configs/tokenizer/tok_g6_v1000.yaml";       name = "fsq_g6_v1000.pt" }   # FSQ-native (8,5,5,5)
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
