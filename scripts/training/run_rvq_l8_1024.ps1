$ErrorActionPreference = "Continue"
$env:PYTHONPATH = "src"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
$busy = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -like '*train_tokenizer*' })
if ($busy.Count -gt 0) {
    Write-Host "ABORT: a tokenizer trainer is already running (PID $($busy[0].ProcessId)). 4GB GPU fits one." -ForegroundColor Red
    exit 1
}
& ".venv\Scripts\python.exe" -u -m text2motion.train.train_tokenizer `
    --config configs/tokenizer/rvq_l8_1024.yaml --tokenizer rvq `
    --epochs 500 --batch_size 128 --eval_every 25 --ckpt_name rvq_l8_1024.pt --resume `
    2>&1 | Tee-Object -FilePath outputs/sweep_rvq_l8_1024.log -Append
