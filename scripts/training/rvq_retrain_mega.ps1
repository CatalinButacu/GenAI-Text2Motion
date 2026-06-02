# Retrain the RVQ tokenizer on richer multi-source data, including Inter-X
# per-person motions. The current best_model.pt was trained on AMASS +
# HumanML3D only -- its codebook hasn't seen two-person interaction-specific
# poses (close contact, mirrored gestures, hand-holding etc.). Resuming on
# mega data lets the codebook adapt without forgetting the existing AMASS
# knowledge.
#
# Backup of the original RVQ already saved at:
#   checkpoints/rvq_tokenizer/best_model_pre_interx_20260515.pt
#
# Hyperparams:
#   --data mega        AMASS + HumanML3D + ARCTIC + InterX (per-person)
#   --max-samples-per-source 2500  caps each source for laptop RAM safety;
#                      total ~7500 samples (8K with HML3D < 2500 maybe)
#   --resume <backup>  start from the existing tokenizer; preserves what we
#                      already learned, just adds new codebook entries
#   --lr 1e-4          half of the original 2e-4 -- safer for fine-tune
#   --epochs 20        enough to adapt the codebook without overfitting
#   --batch-size 32    matches original training
#
# Wall-time estimate at ~6000 train samples / 32 batch / 3 sec/step:
#   ~190 batches/epoch -> ~10 min/epoch -> 20 epochs ~ 3.3 hours.

$ts  = Get-Date -Format 'yyyyMMdd-HHmmss'
$out = "rvq_retrain_$ts.out"
$err = "rvq_retrain_$ts.err"

$argList = @(
    "scripts/training/train_rvq_tokenizer.py",
    "--data", "mega",
    "--max-samples-per-source", "2500",
    "--resume", "checkpoints/rvq_tokenizer/best_model_pre_interx_20260515.pt",
    "--warm-start",
    # Use the SAME shared stats the prior RVQ trained against. This both
    # (1) keeps the z-norm reference identical so fine-tuning doesn't shift
    # the codebook into a new normalization frame, and (2) skips the in-RAM
    # computeMotionStats() over ~1.6M frames which OOM'd on first attempt.
    "--stats-path", "data/stats/amass_full.npz",
    "--trans-stats-path", "data/stats/translation_per_source.npz",
    "--epochs", "20",
    "--batch-size", "32",
    "--lr", "1e-4",
    "--num-workers", "0",
    "--reset-dead-every", "5",
    "--device", "cuda"
)

Write-Host "--- RVQ retrain on mega (incl. InterX per-person) ---"
Write-Host "  resume: best_model_pre_interx_20260515.pt"
Write-Host "  data: mega (AMASS + HumanML3D + ARCTIC + Inter-X)"
Write-Host "  cap:  2500 samples per source"
Write-Host "  20 epochs / batch=32 / lr=1e-4 (fine-tune scale)"
Write-Host "  estimated wall time ~ 3.3 hours"
Write-Host ""
Write-Host "  stdout -> $out"
Write-Host "  stderr -> $err"

$p = Start-Process -FilePath python `
    -ArgumentList $argList `
    -RedirectStandardOutput $out `
    -RedirectStandardError $err `
    -WindowStyle Hidden `
    -PassThru

$p.Id | Out-File -Encoding utf8 rvq_retrain.pid

Write-Host ""
Write-Host "Launched PID=$($p.Id)"
