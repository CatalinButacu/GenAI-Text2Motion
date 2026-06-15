# Tokenizer latent-space sweep driver. Runs each variant sequentially at the SAME budget as the
# done anchors (500 ep, eval every 25), resume-safe, manifest-logged, never clobbering a winner.
# Variants chosen for balance (2 fundamental axes, not a giant grid): group count {4,6,8} and
# per-group vocab {512,1000,2560}. Anchors already done: g6_v1000=0.0266, g6_v512=0.0307, RVQ=0.0382.
$ErrorActionPreference = "Stop"
$env:PYTHONPATH = "src"
$py = ".venv\Scripts\python.exe"

$variants = @(
    @{ cfg = "configs/tok_g4_v1000.yaml";  name = "tok_g4_v1000.pt" },
    @{ cfg = "configs/tok_g8_v1000.yaml";  name = "tok_g8_v1000.pt" },
    @{ cfg = "configs/tok_g6_v2560.yaml";  name = "tok_g6_v2560.pt" }
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
"=== SWEEP COMPLETE ===" | Out-File -Append outputs/sweep_done.txt -Encoding utf8
