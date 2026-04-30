#!/usr/bin/env bash
# =============================================================================
# AWS GPU training wrapper — keeps spend bounded, never leaves resources up.
#
# Lifecycle:
#   ./run.sh up      -> provisions EC2 + starts training (asks confirm)
#   ./run.sh logs    -> tails training logs over SSH
#   ./run.sh wait    -> blocks until self-terminate, then sync + destroy
#   ./run.sh sync    -> pulls checkpoints from S3 to local
#   ./run.sh down    -> emergency: force-destroy all terraform resources
#   ./run.sh status  -> shows EC2 instance state
# =============================================================================
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

REGION_DEFAULT="eu-west-1"

readTfvar() {
  # readTfvar <key> <default>: extract a string value from terraform.tfvars
  local key="$1" default="${2:-}"
  if [ -f terraform.tfvars ]; then
    awk -F= -v k="$key" '
      $1 ~ "^[[:space:]]*"k"[[:space:]]*$" {
        gsub(/^[[:space:]]+|[[:space:]]+$/, "", $2)
        gsub(/^"|"$/, "", $2)
        print; exit
      }' terraform.tfvars | tr -d '\r'
  else
    echo "$default"
  fi
}

REGION="$(readTfvar region "$REGION_DEFAULT")"
S3_BUCKET="$(readTfvar s3_bucket "")"

confirm() {
  local prompt="$1"
  read -r -p "$prompt [y/N] " reply
  [[ "$reply" =~ ^[Yy]$ ]]
}

requireTfState() {
  if [ ! -f terraform.tfstate ] && [ ! -f .terraform/terraform.tfstate ]; then
    echo "no terraform state found — run './run.sh up' first" >&2
    return 1
  fi
}

instanceId() {
  terraform output -raw instance_id 2>/dev/null || true
}

cmdUp() {
  if [ ! -d .terraform ]; then
    echo "Initializing terraform..."
    terraform init -input=false
  fi

  echo
  echo "About to provision:"
  echo "  region   : $REGION"
  echo "  instance : g4dn.xlarge spot (T4 16 GB, ~\$0.16/hr)"
  echo "  budget   : ~\$1-3 for one full training run (RVQ + MotionSSM)"
  echo "  cleanup  : EC2 self-terminates after training; run './run.sh wait' to also destroy IAM/SG/etc."
  echo

  if ! confirm "Apply terraform now?"; then
    echo "Aborted."
    return 1
  fi

  terraform apply -auto-approve

  echo
  echo "Provisioned. Suggested next steps:"
  echo "  ./run.sh logs    # watch training (Ctrl+C to detach, training keeps running)"
  echo "  ./run.sh wait    # block until self-terminate, then auto sync + destroy"
}

cmdStatus() {
  requireTfState
  local id; id="$(instanceId)"
  if [ -z "$id" ]; then
    echo "no instance in current state"
    return 0
  fi
  aws ec2 describe-instances --instance-ids "$id" --region "$REGION" \
    --query 'Reservations[0].Instances[0].[State.Name,InstanceType,LaunchTime]' \
    --output table
}

cmdLogs() {
  requireTfState
  local cmd; cmd="$(terraform output -raw watch_logs)"
  echo "$ $cmd"
  echo "(Ctrl+C detaches; the EC2 instance keeps running.)"
  eval "$cmd"
}

cmdWait() {
  requireTfState
  local id; id="$(instanceId)"
  if [ -z "$id" ]; then
    echo "no instance in current state — nothing to wait for"
    return 0
  fi

  echo "Waiting for instance $id to self-terminate (this is normal — startup.sh terminates after training)..."
  aws ec2 wait instance-terminated --instance-ids "$id" --region "$REGION"
  echo "Instance terminated."

  if [ -n "$S3_BUCKET" ]; then
    echo "Syncing checkpoints from s3://$S3_BUCKET/checkpoints -> ../../../checkpoints/"
    mkdir -p ../../../checkpoints
    aws s3 sync "s3://$S3_BUCKET/checkpoints" ../../../checkpoints
  else
    echo "WARN: s3_bucket not in terraform.tfvars — skipping sync. Run './run.sh sync' manually."
  fi

  echo "Destroying remaining terraform-managed resources (IAM, SG, etc.)..."
  terraform destroy -auto-approve
  echo "All resources destroyed. You're not paying for anything."
}

cmdSync() {
  if [ -z "$S3_BUCKET" ]; then
    echo "ERROR: s3_bucket not in terraform.tfvars" >&2
    return 1
  fi
  mkdir -p ../../../checkpoints
  aws s3 sync "s3://$S3_BUCKET/checkpoints" ../../../checkpoints
}

cmdDown() {
  requireTfState
  echo "Force-destroy ALL terraform-managed AWS resources for this stack."
  if ! confirm "Confirm destroy?"; then
    echo "Aborted."
    return 1
  fi
  terraform destroy -auto-approve
}

cmdHelp() {
  cat <<'EOF'
AWS GPU training wrapper

Subcommands:
  up      Provision EC2 spot instance and start training (asks confirmation).
  status  Show current instance state.
  logs    SSH and tail /var/log/user-data.log on the instance.
  wait    Block until self-terminate, then sync checkpoints + destroy.
  sync    Download checkpoints from S3 to ./checkpoints (idempotent).
  down    Emergency: force-destroy everything (asks confirmation).
  help    This message.

Typical run:
  ./run.sh up
  ./run.sh logs        # watch ~3-6 hours of training
  ./run.sh wait        # auto-sync checkpoints + destroy when done
EOF
}

case "${1:-help}" in
  up)     cmdUp ;;
  status) cmdStatus ;;
  logs)   cmdLogs ;;
  wait)   cmdWait ;;
  sync)   cmdSync ;;
  down)   cmdDown ;;
  help|-h|--help) cmdHelp ;;
  *) echo "unknown subcommand: $1" >&2; cmdHelp; exit 1 ;;
esac
