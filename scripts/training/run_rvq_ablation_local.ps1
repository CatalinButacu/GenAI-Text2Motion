# powershell -ExecutionPolicy Bypass -File scripts/training/run_rvq_ablation_local.ps1 -BudgetHours 8
param([double]$BudgetHours = 8, [int]$StaleMinutes = 45)
$ErrorActionPreference = "Stop"
Import-Module (Join-Path $PSScriptRoot "TrainingRun.psm1") -Force
$null = Enter-RepoRoot -StartDirectory $PSScriptRoot
$start = Get-Date
$config = "configs/generator/gen_pilot_rvq8x1024.yaml"
$tokenizer = "artifacts/checkpoints/tokenizer/rvq_l8_1024.pt"
$pretrained = "artifacts/checkpoints/generator/generator_transformer_rvq_pretrained.pt"
$pretrainResume = "artifacts/checkpoints/generator/generator_transformer_rvq_pretrained_last.pt"
$finetuneResume = "artifacts/checkpoints/generator/generator_transformer_rvq_bs8_last.pt"
$gate = "artifacts/outputs/gates/transformer_rvq_34m_overfit.json"
Write-Host "=== transformer+RVQ downstream-generation ablation start $(Get-Date -Format u)  budget ${BudgetHours}h ==="

function Get-HoursLeft {
    return [math]::Round($BudgetHours - ((Get-Date) - $start).TotalHours, 2)
}

function Stop-OnFailure([int]$code, [string]$stage) {
    if ($code -ne 0) {
        Write-Host "ABLATION STOPPED at ${stage} (exit code $code); no DONE marker"
        exit $code
    }
}

if (-not (Test-RunComplete -ResumePath $pretrainResume -Epochs 20)) {
    Stop-OnFailure (Invoke-TrainingStep -Tag "pretrain_transformer_rvq" -LogFile "artifacts/outputs/rvq_ablation_pretrain.log" `
            -MaxHours (Get-HoursLeft) -StaleMinutes $StaleMinutes -Arguments @(
                "-m", "text2motion.app.cli", "pretrain", "--config", $config, "--backbone", "transformer",
                "--token_pack", "data/amass_tokens_rvq8x1024.npz", "--epochs", "20", "--batch_size", "64",
                "--ckpt", $pretrained, (Get-ResumeFlag -ResumePath $pretrainResume)
            )) "pretrain"
}

if (-not (Test-Path $gate)) {
    Stop-OnFailure (Invoke-TrainingStep -Tag "gate_transformer_rvq" -LogFile "artifacts/outputs/rvq_ablation_gate.log" `
            -MaxHours 2 -Arguments @(
                "-m", "text2motion.app.cli", "sanity-overfit", "--config", $config,
                "--backbone", "transformer", "--tokenizer_ckpt", $tokenizer, "--overfit_gate", $gate
            )) "gate"
}

if (-not (Test-RunComplete -ResumePath $finetuneResume -Epochs 18)) {
    Stop-OnFailure (Invoke-TrainingStep -Tag "train_transformer_rvq" -LogFile "artifacts/outputs/rvq_ablation_finetune.log" `
            -MaxHours (Get-HoursLeft) -StaleMinutes $StaleMinutes -Arguments @(
                "-m", "text2motion.app.cli", "train-generator", "--config", $config, "--backbone", "transformer",
                "--tokenizer_ckpt", $tokenizer, "--overfit_gate", $gate, "--init_ckpt", $pretrained,
                "--ckpt_name", "generator_transformer_rvq_bs8.pt", "--epochs", "18", "--batch_size", "8",
                "--num_workers", "4", "--eval_every", "6", "--cfg_scale", "6.0", "--temperature", "1.0",
                (Get-ResumeFlag -ResumePath $finetuneResume)
            )) "fine-tune"
}

$null = Write-DoneMarker -Path "artifacts/outputs/RVQ_ABLATION_DONE.txt" -ExitCodes @(0)
