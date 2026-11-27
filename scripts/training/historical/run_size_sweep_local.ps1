# powershell -ExecutionPolicy Bypass -File scripts/training/historical/run_size_sweep_local.ps1 -Widths 320 -Epochs 60 -ValCePatience 6 -Tag sweep
param([double]$BudgetHoursPerRun = 6, [int]$StaleMinutes = 45, [int]$Epochs = 20, [int]$ValCePatience = 6,
    [string[]]$Widths = @("320"), [string]$Tag = "sweep", [switch]$Pilot, [switch]$ResumePilot)
$ErrorActionPreference = "Stop"
Import-Module (Join-Path $PSScriptRoot "../TrainingRun.psm1") -Force
$null = Enter-RepoRoot -StartDirectory $PSScriptRoot
$tokenizer = "runs/checkpoints/tokenizer/fsq_g8_v1024.pt"
Write-Host "=== size sweep start $(Get-Date -Format u) ==="
if ($ResumePilot -and -not $Pilot) {
    throw "-ResumePilot requires -Pilot"
}

function __GatePath([string]$width, [string]$backbone) {
    $prefix = if ($Pilot) { $Tag } else { "size_sweep" }
    return "runs/outputs/gates/${prefix}_${backbone}_w${width}_overfit.json"
}

function Invoke-SweepGate([string]$width, [string]$backbone) {
    $gate = __GatePath $width $backbone
    if (Test-Path $gate) {
        Write-Host "SKIP gate $backbone w$width ($gate exists)"
        return 0
    }
    $logPrefix = if ($Pilot) { "${Tag}_gate" } else { "size_sweep_gate" }
    return Invoke-TrainingStep -Tag "gate_${backbone}_w$width" -LogFile "runs/outputs/${logPrefix}_${backbone}_w$width.log" `
        -MaxHours 1 -StreamLogs:$Pilot -PollSeconds 5 -Arguments @(
            "-m", "text2motion.app.cli", "sanity-overfit", "--config", "configs/generator/size_sweep/width_$width.yaml",
            "--backbone", $backbone, "--tokenizer_ckpt", $tokenizer, "--overfit_gate", $gate
        )
}

function Invoke-SweepRun([string]$width, [string]$backbone) {
    $name = "generator_${backbone}_w${width}_${Tag}.pt"
    $resume = "runs/checkpoints/generator/$([IO.Path]::GetFileNameWithoutExtension($name))_last.pt"
    if ((Test-Path $resume) -and -not $ResumePilot) {
        Write-Host "SKIP run $backbone w$width ($resume exists; a sweep run is never resumed)"
        return 0
    }
    $staleCode = if ($Pilot) { 125 } else { 124 }
    $logFile = "runs/outputs/${Tag}_${backbone}_w$width.log"
    if ($ResumePilot -and (Test-Path $resume) -and -not (Test-Path $logFile)) {
        throw "recovery checkpoint exists without the original budget log: $resume"
    }
    $remainingHours = $BudgetHoursPerRun
    $resumeFlag = ""
    if ($ResumePilot -and (Test-Path $logFile)) {
        $elapsed = ((Get-Date).ToUniversalTime() - (Get-Item $logFile).CreationTimeUtc).TotalHours
        $remainingHours -= $elapsed
        if ($remainingHours -le 0) {
            Write-Warning "PARTIAL: $backbone w$width has no time remaining from its original cap"
            return 124
        }
        $resumeFlag = Get-ResumeFlag -ResumePath $resume
        if (-not $resumeFlag) {
            throw "recovery requires the interrupted pilot's durable checkpoint: $resume"
        }
        $stamp = (Get-Date).ToUniversalTime().ToString("yyyyMMddTHHmmssZ")
        $logFile = "runs/outputs/${Tag}_${backbone}_w${width}_resume_${stamp}.log"
        Write-Host "RECOVERY: $backbone w$width has $remainingHours h remaining; original logs preserved"
    }
    return Invoke-TrainingStep -Tag "${Tag}_${backbone}_w$width" -LogFile $logFile `
        -MaxHours $remainingHours -StaleMinutes $StaleMinutes -StaleExitCode $staleCode -StreamLogs:$Pilot -PollSeconds 5 -Arguments @(
            "-m", "text2motion.app.cli", "train-generator", "--config", "configs/generator/size_sweep/width_$width.yaml",
            "--backbone", $backbone, "--tokenizer_ckpt", $tokenizer,
            "--overfit_gate", (__GatePath $width $backbone),
            "--ckpt_name", $name, "--epochs", "$Epochs", "--val_ce_patience", "$ValCePatience",
            "--batch_size", "8", "--num_workers", "0", "--eval_every", "2",
            "--cfg_scale", "6.0", "--temperature", "1.0", $resumeFlag
        )
}

$codes = @()
if ($Pilot -and -not $ResumePilot) {
    $previous = @(Get-ChildItem "runs/outputs" -File -Filter "${Tag}*")
    foreach ($width in $Widths) {
        foreach ($backbone in @("transformer", "mamba")) {
            $previous += @(Get-ChildItem "runs/checkpoints/generator" -File -Filter "generator_${backbone}_w${width}_${Tag}*")
            if (Test-Path (__GatePath $width $backbone)) {
                throw "pilot gate already exists; choose a fresh Tag: $Tag"
            }
        }
    }
    if ($previous.Count -gt 0) {
        throw "pilot outputs already exist; choose a fresh Tag: $Tag"
    }
}
if ($Pilot) {
    foreach ($width in $Widths) {
        foreach ($backbone in @("transformer", "mamba")) {
            if ($ResumePilot -and -not (Test-Path (__GatePath $width $backbone))) {
                throw "recovery requires the original passed gate for $backbone w$width"
            }
            $code = Invoke-SweepGate $width $backbone
            if ($code -ne 0) {
                Write-Error "STOP: gate $backbone w$width failed with code $code; no training started"
                exit 1
            }
        }
    }
}
foreach ($width in $Widths) {
    foreach ($backbone in @("transformer", "mamba")) {
        if (-not $Pilot) {
            $code = Invoke-SweepGate $width $backbone
            $codes += $code
            if ($code -ne 0) {
                Write-Host "STOP: gate $backbone w$width failed with code $code"
                exit 1
            }
        }
        $code = Invoke-SweepRun $width $backbone
        $codes += $code
        if ($Pilot -and $code -eq 124) {
            Write-Warning "PARTIAL: $backbone w$width reached its ${BudgetHoursPerRun}h cap; continuing the paired pilot, not claiming convergence"
            continue
        }
        if ($code -ne 0) {
            Write-Host "STOP: run $backbone w$width failed with code $code"
            exit 1
        }
    }
}

if ($Pilot -and $codes -contains 124) {
    "partial $((Get-Date).ToUniversalTime().ToString('u')); exit codes $($codes -join ','); time capped, not converged" |
        Out-File "runs/outputs/$($Tag.ToUpper())_PARTIAL.txt" -Encoding utf8
    Write-Warning "Pilot finished with time-capped runs; no DONE marker"
    exit 124
}
if (-not (Write-DoneMarker -Path "runs/outputs/$($Tag.ToUpper())_DONE.txt" -ExitCodes $codes)) {
    exit 1
}
