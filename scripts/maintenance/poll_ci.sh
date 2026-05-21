#!/usr/bin/env bash
# Poll GitHub Actions for a specific commit SHA until the run completes.
# Usage: poll_ci.sh <owner/repo> <sha>
set -euo pipefail

REPO=${1:?need owner/repo}
SHA=${2:?need commit sha}

while true; do
  payload=$(curl -fsS "https://api.github.com/repos/${REPO}/actions/runs?head_sha=${SHA}&per_page=1")
  status=$(python -c "import json,sys; d=json.loads(sys.argv[1]); r=d['workflow_runs']; print(f\"{r[0]['status']}|{r[0]['conclusion']}|{r[0]['html_url']}\") if r else print('pending||')" "$payload")
  echo "[$(date +%H:%M:%S)] $status"
  case "$status" in
    completed*) break ;;
  esac
  sleep 10
done
