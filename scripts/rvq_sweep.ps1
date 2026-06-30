# RVQ hypertune sweep — find the best strong-RVQ config (levels x codebook) by recon-FID.
# Seed 2026 (project default), 500 ep, eval every 25, resume-safe, manifest-logged, ckpt per config.
# Levels {4,6,8} x codebook {512(,1024)}; enc/dec held constant. rvq_l6_512 doubles as the
# foundational baseline manifest re-run.
$ErrorActionPreference = "Continue"  # native stderr via 2>&1 must not terminate the driver
$env:PYTHONPATH = "src"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
$py = ".venv\Scripts\python.exe"

$variants = @(
    @{ cfg = "configs/tokenizer/rvq_l6_512.yaml";  name = "rvq_l6_512.pt" },
    @{ cfg = "configs/tokenizer/rvq_l4_512.yaml";  name = "rvq_l4_512.pt" },
    @{ cfg = "configs/tokenizer/rvq_l8_512.yaml";  name = "rvq_l8_512.pt" },
    @{ cfg = "configs/tokenizer/rvq_l6_1024.yaml"; name = "rvq_l6_1024.pt" }
)

foreach ($v in $variants) {
    $stem = [System.IO.Path]::GetFileNameWithoutExtension($v.name)
    $log = "outputs/sweep_$stem.log"
    "=== $(Get-Date -Format o)  training $stem ===" | Out-File -Append $log -Encoding utf8
    & $py -u -m text2motion.train.train_tokenizer --config $v.cfg --tokenizer rvq `
        --epochs 500 --batch_size 128 --eval_every 25 --ckpt_name $v.name --resume `
        2>&1 | Out-File -Append $log -Encoding utf8
    "=== $(Get-Date -Format o)  done $stem ===" | Out-File -Append $log -Encoding utf8
}
"=== RVQ SWEEP COMPLETE ===" | Out-File outputs/rvq_sweep_done.txt -Encoding utf8
