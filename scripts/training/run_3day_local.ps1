# powershell -ExecutionPolicy Bypass -File scripts/training/run_3day_local.ps1 -BudgetHours 70
param([double]$BudgetHours = 70, [int]$StaleMinutes = 45)
$ErrorActionPreference = "Stop"
Import-Module (Join-Path $PSScriptRoot "TrainingRun.psm1") -Force
$null = Enter-RepoRoot -StartDirectory $PSScriptRoot
$start = Get-Date
$fsqConfig = "configs/generator/gen_pilot_fsq8x1024.yaml"
$fsqTokenizer = "artifacts/checkpoints/tokenizer/fsq_g8_v1024.pt"
Write-Host "=== 3-day local chain start $(Get-Date -Format u)  budget ${BudgetHours}h ==="

function Get-HoursLeft {
    return [math]::Round($BudgetHours - ((Get-Date) - $start).TotalHours, 1)
}

function Invoke-WithRetries([string]$tag, [string]$resume, [int]$epochs, [int]$tries, [string[]]$arguments) {
    for ($try = 1; $try -le $tries; $try++) {
        if (Test-RunComplete -ResumePath $resume -Epochs $epochs) {
            Write-Host "SKIP $tag (all $epochs epochs are in $resume)"
            return 0
        }
        $left = Get-HoursLeft
        if ($left -le 1) {
            Write-Host "STOP ${tag}: budget exhausted"
            return 1
        }
        $code = Invoke-TrainingStep -Tag "$tag (try $try)" -LogFile "artifacts/outputs/$tag.log" `
            -MaxHours $left -StaleMinutes $StaleMinutes `
            -Arguments ($arguments + @(Get-ResumeFlag -ResumePath $resume))
        if ($code -ne 124) {
            return $code
        }
    }
    return 124
}

function Invoke-Gate([string]$backbone, [string]$gate) {
    if (Test-Path $gate) {
        return 0
    }
    return Invoke-TrainingStep -Tag "gate_$backbone" -LogFile "artifacts/outputs/gate_$backbone.log" `
        -MaxHours 2 -Arguments @(
            "-m", "text2motion.app.cli", "sanity-overfit", "--config", $fsqConfig,
            "--backbone", $backbone, "--tokenizer_ckpt", $fsqTokenizer, "--overfit_gate", $gate
        )
}

function Stop-OnFailure([int]$code, [string]$phase) {
    if ($code -ne 0) {
        Write-Host "CHAIN STOPPED at ${phase} (exit code $code); no DONE marker"
        exit $code
    }
}

Stop-OnFailure (Invoke-WithRetries "phase1_isovocab" "artifacts/checkpoints/tokenizer/tokenizer_isovocab_last.pt" 50 1 @(
        "-m", "text2motion.app.cli", "train-tokenizer", "--config", "configs/tokenizer/tokenizer_isovocab.yaml",
        "--tokenizer", "fsq", "--ckpt_name", "tokenizer_isovocab.pt", "--epochs", "50", "--eval_every", "5"
    )) "phase 1"

Stop-OnFailure (Invoke-Gate "transformer" "artifacts/outputs/gates/transformer_34m_overfit.json") "phase 2 gate"
Stop-OnFailure (Invoke-WithRetries "phase2_transformer_bs8" "artifacts/checkpoints/generator/generator_transformer_bs8_last.pt" 18 1 @(
        "-m", "text2motion.app.cli", "train-generator", "--config", $fsqConfig, "--backbone", "transformer",
        "--tokenizer_ckpt", $fsqTokenizer, "--overfit_gate", "artifacts/outputs/gates/transformer_34m_overfit.json",
        "--init_ckpt", "artifacts/checkpoints/generator/generator_transformer_pretrained.pt",
        "--ckpt_name", "generator_transformer_bs8.pt", "--epochs", "18", "--batch_size", "8",
        "--eval_every", "6", "--cfg_scale", "6.0", "--temperature", "1.0"
    )) "phase 2"

Stop-OnFailure (Invoke-WithRetries "phase3_mamba_pretrain" "artifacts/checkpoints/generator/generator_mamba_pretrained_last.pt" 20 6 @(
        "-m", "text2motion.app.cli", "pretrain", "--config", $fsqConfig, "--backbone", "mamba",
        "--token_pack", "data/amass_tokens_fsq8x1024.npz", "--epochs", "20", "--batch_size", "8",
        "--grad_accum", "8", "--ckpt", "artifacts/checkpoints/generator/generator_mamba_pretrained.pt"
    )) "phase 3"

Stop-OnFailure (Invoke-Gate "mamba" "artifacts/outputs/gates/mamba_34m_overfit.json") "phase 4 gate"
Stop-OnFailure (Invoke-WithRetries "phase4_mamba_finetune" "artifacts/checkpoints/generator/generator_mamba_bs8_last.pt" 18 4 @(
        "-m", "text2motion.app.cli", "train-generator", "--config", $fsqConfig, "--backbone", "mamba",
        "--tokenizer_ckpt", $fsqTokenizer, "--overfit_gate", "artifacts/outputs/gates/mamba_34m_overfit.json",
        "--init_ckpt", "artifacts/checkpoints/generator/generator_mamba_pretrained.pt",
        "--ckpt_name", "generator_mamba_bs8.pt", "--epochs", "18", "--batch_size", "4", "--grad_accum", "2",
        "--eval_every", "6", "--cfg_scale", "6.0", "--temperature", "1.0"
    )) "phase 4"

$null = Write-DoneMarker -Path "artifacts/outputs/CHAIN_DONE.txt" -ExitCodes @(0)
