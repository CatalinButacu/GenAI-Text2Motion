# NEUTRALIZED 2026-06-17. This driver caused a duplicate-instance GPU collision (see
# paper/tokenizer-matrix-status.md). The harness was respawning it as a supervised background task.
# Replaced with an immediate no-op so any respawn exits harmlessly. Remaining matrix work now lives
# in scripts/matrix_remaining.ps1 -- run THAT yourself in a normal terminal, never as a bg task.
"$(Get-Date -Format o)  matrix_complete.ps1 is neutralized; use scripts/matrix_remaining.ps1" |
    Out-File -Append outputs/matrix_complete_neutralized.txt -Encoding utf8
exit 0
