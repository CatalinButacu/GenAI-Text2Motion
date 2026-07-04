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
