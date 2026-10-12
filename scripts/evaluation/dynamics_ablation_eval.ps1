$ErrorActionPreference = "Continue"
$root = $PSScriptRoot
while ($root -and -not (Test-Path (Join-Path $root "pyproject.toml"))) {
    $root = Split-Path $root -Parent
}
Set-Location $root
$env:PYTHONPATH = "src"
$env:HF_HUB_OFFLINE = "1"
$py = ".\.venv\Scripts\python.exe"
$config = "configs/generator/gen_pilot_fsq8x1024.yaml"
$tokenizer = "artifacts/checkpoints/tokenizer/fsq_g8_v1024.pt"
$checkpoints = "artifacts/checkpoints/generator"
$failed = @()
New-Item -ItemType Directory -Force artifacts/outputs/qa | Out-Null
Remove-Item artifacts/outputs/DYNAMICS_ABLATION_EVAL_DONE.txt -ErrorAction SilentlyContinue
Write-Output "=== dynamics ablation evaluation start $(Get-Date -Format u) (HF offline) ==="

function Test-Result([string]$logFile, [string]$pattern) {
    return (Test-Path $logFile) -and (Select-String -Path $logFile -Pattern $pattern -Quiet)
}

foreach ($backbone in @("transformer", "mamba")) {
    $ckpt = "$checkpoints/generator_${backbone}_bs8_dyn_e72.pt"
    $logFile = "artifacts/outputs/dyn_test_$backbone.log"
    if (Test-Result $logFile "clips 2189  FID") {
        Write-Output "SKIP test evaluation $backbone (result already in $logFile -- test is touched once)"
        continue
    }
    Write-Output "`n=== test evaluation: $backbone dyn_e72 ==="
    & $py -u -m text2motion.app.cli evaluate `
        --config $config --backbone $backbone --ckpt $ckpt --tokenizer_ckpt $tokenizer `
        --split test --cfg_scale 6.0 --temperature 1.0 --length_mode fixed `
        --mm_clips 100 --mm_repeats 30 *>&1 | Out-File $logFile -Encoding utf8
    if (-not (Test-Result $logFile "clips 2189  FID")) {
        $failed += "test evaluation $backbone"
    }
}

foreach ($backbone in @("transformer", "mamba")) {
    foreach ($variant in @("e60", "dyn_e72")) {
        $ckpt = "$checkpoints/generator_${backbone}_bs8_$variant.pt"
        $logFile = "artifacts/outputs/qa/naturalness_${backbone}_$variant.log"
        Write-Output "`n=== naturalness: $backbone $variant ==="
        & $py -u scripts/qa/motion_naturalness_probe.py --config $config --backbone $backbone `
            --ckpt $ckpt --tokenizer_ckpt $tokenizer --clips 64 --mesh_clips 0 `
            --out "artifacts/outputs/qa/naturalness_${backbone}_$variant.json" *>&1 |
            Out-File $logFile -Encoding utf8
        if (-not (Test-Result $logFile "^generated ")) {
            $failed += "naturalness $backbone $variant"
        }
    }
}

if ($failed.Count -gt 0) {
    Write-Output "`n=== FAILED: $($failed -join '; ') -- see the step logs; no DONE marker written ==="
} else {
    Write-Output "`n=== all steps succeeded $(Get-Date -Format u) ==="
    "done $(Get-Date -Format u)" | Out-File artifacts/outputs/DYNAMICS_ABLATION_EVAL_DONE.txt -Encoding utf8
}
