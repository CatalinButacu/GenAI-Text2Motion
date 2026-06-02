# Resume MotionSSM from the all3rd-50e best checkpoint with a 30x-lower LR.
# Warm-start drops the saved optimizer + OneCycleLR state so --lr 1e-5 actually wins.
# Output: training metrics go to checkpoints/motion_ssm/<new_run_id>/training.log
#         python stdout/stderr go to checkpoints/motion_ssm/launcher_<ts>.{out,err}

$ts  = Get-Date -Format 'yyyyMMdd-HHmmss'
$out = "checkpoints/motion_ssm/launcher_$ts.out"
$err = "checkpoints/motion_ssm/launcher_$ts.err"

$argList = @(
  "scripts/training/train_motion_ssm.py",
  "--data-source", "unified", "--sources", "humanml3d",
  "--use-sbert", "--bidirectional", "--use-film", "--gradient-checkpointing",
  "--d-model", "384", "--d-state", "64", "--n-layers", "6",
  "--max-motion-length", "200",
  "--batch-size", "64",
  "--lr", "1e-5",
  "--epochs", "50",
  "--num-workers", "4",
  "--checkpoint-dir", "checkpoints/motion_ssm",
  "--rvq-checkpoint", "checkpoints/rvq_tokenizer/best_model.pt",
  "--resume", "checkpoints/motion_ssm/20260507-175255-all3rd-50e/best_model.pt",
  "--warm-start",
  "--device", "cuda"
)

$p = Start-Process -FilePath python -ArgumentList $argList `
        -RedirectStandardOutput $out -RedirectStandardError $err `
        -WindowStyle Hidden -PassThru

$p.Id | Out-File -Encoding utf8 motion_ssm_resume.pid
"Launched PID=$($p.Id)  out=$out  err=$err  pidfile=motion_ssm_resume.pid"
