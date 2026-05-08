#!/usr/bin/env bash
# =============================================================================
# run.sh — wrapper around Terraform + AWS CLI for the GPU training lifecycle
# =============================================================================
#
# Subcommands (typed like `./run.sh up`):
#
#   up        Provision EC2 spot + IAM + security group, start training.
#   status    Show "is the EC2 still running?".
#   logs      SSH and follow /var/log/user-data.log on the EC2.
#   monitor   Poll every 5 min, print one line per check (boot + train).
#   wait      Block until EC2 self-terminates, then sync + cleanup IAM/SG.
#   sync      Download trained checkpoints from S3 to ./checkpoints.
#   down      Emergency: force-destroy ALL terraform-managed resources.
#   help      This message.
#
# Costs:
#   `up` starts a g4dn.xlarge spot at ~$0.28/hr.  Full training = ~$1.70.
#   Nothing else costs money on its own (status/logs/monitor are free queries).
#
# Read CLOUD-GUIDE.md alongside this file for an explanation of *what*
# AWS, S3, Terraform, IAM, etc. actually are.
# =============================================================================

# Bash strict mode (errors out on first failure, no silent bugs)
#   -e   exit on any failure
#   -u   error on undefined variables (catches typos)
#   -o pipefail  if a piped command fails, the whole pipe fails
set -euo pipefail

# Always run from this script's directory (so relative paths work even if
# the user calls it from somewhere else)
cd "$(dirname "${BASH_SOURCE[0]}")"

# -----------------------------------------------------------------------------
# Read configuration values from terraform.tfvars (the file you edit by hand
# to set region, instance type, bucket name, etc.). We need these in this
# wrapper too — for SSH, S3 paths, etc.
# -----------------------------------------------------------------------------
readTfvar() {
  # Tiny awk parser: given a key like "region", find a line of the form
  #   region = "eu-west-1"
  # and print just `eu-west-1`. Returns the default if not found.
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

REGION="$(readTfvar region eu-west-1)"
S3_BUCKET="$(readTfvar s3_bucket "")"
KEY_PAIR="$(readTfvar key_pair_name dissertation-${REGION})"
PEM_PATH="$HOME/.ssh/${KEY_PAIR}.pem"

# -----------------------------------------------------------------------------
# Tiny helpers
# -----------------------------------------------------------------------------
confirm() {
  # "Are you sure?" prompt — returns 0 if user types y/Y, else non-zero
  local prompt="$1"
  read -r -p "$prompt [y/N] " reply
  [[ "$reply" =~ ^[Yy]$ ]]
}

requireTfState() {
  # Most subcommands need to know "is there an EC2 currently provisioned?"
  # Terraform answers via terraform.tfstate. If that file is missing, no
  # resources have been created.
  if [ ! -f terraform.tfstate ] && [ ! -f .terraform/terraform.tfstate ]; then
    echo "ERROR: no terraform state — run './run.sh up' first" >&2

    return 1
  fi
}

instanceId() {
  # Terraform records outputs (declared in main.tf). We can read them with
  # `terraform output -raw <name>`. Returns empty if the EC2 doesn't exist.
  terraform output -raw instance_id 2>/dev/null || true
}

publicIp() {
  terraform output -raw public_ip 2>/dev/null || true
}

# =============================================================================
# Subcommand: up — provision everything and start training
# =============================================================================
cmdUp() {
  # `terraform init` downloads the AWS provider plugin. Idempotent — only
  # actually downloads on first run or after Terraform version changes.
  if [ ! -d .terraform ]; then
    echo "First-time setup: terraform init..."
    terraform init -input=false
  fi

  echo
  echo "About to provision in region $REGION:"
  echo "  • IAM role           (EC2's permission to read/write S3 + self-terminate)"
  echo "  • Security group     (firewall: open SSH port 22 to anywhere)"
  echo "  • EC2 spot instance  (the actual GPU machine)"
  echo
  echo "Cost: ~\$0.28/hr while running. Training takes ~6 hours = ~\$1.70."
  echo "EC2 self-terminates when training is done — your bill stops automatically."
  echo

  if ! confirm "Apply terraform now?"; then
    echo "Aborted."

    return 1
  fi

  # `-auto-approve` skips Terraform's own "are you sure?" prompt (we already
  # asked above). Output shows each resource being created in real time.
  terraform apply -auto-approve

  echo
  echo "Done. The EC2 is now booting. Suggested next steps:"
  echo "  ./run.sh logs       # tail boot log + see training start"
  echo "  ./run.sh monitor    # one update every 5 min until done"
  echo "  ./run.sh wait       # block until done, then auto-cleanup"
}

# =============================================================================
# Subcommand: status — is the EC2 still alive?
# =============================================================================
cmdStatus() {
  requireTfState
  local id; id="$(instanceId)"

  if [ -z "$id" ]; then
    echo "no instance in current state"

    return 0
  fi

  # `aws ec2 describe-instances` queries the EC2 API for instance metadata.
  # `--query` uses JMESPath to extract just the fields we care about.
  aws ec2 describe-instances --instance-ids "$id" --region "$REGION" \
    --query 'Reservations[0].Instances[0].[State.Name,InstanceType,LaunchTime]' \
    --output table
}

# =============================================================================
# Subcommand: logs — tail the EC2's startup log over SSH
# =============================================================================
cmdLogs() {
  requireTfState
  local ip; ip="$(publicIp)"

  if [ -z "$ip" ]; then
    echo "no instance public ip"

    return 1
  fi

  echo "ssh -i $PEM_PATH ubuntu@$ip 'sudo tail -f /var/log/user-data.log'"
  echo "(Ctrl+C detaches; the EC2 keeps running.)"

  # `sudo tail -f` follows the log forever. The EC2 wrote it as root.
  ssh -o StrictHostKeyChecking=no -i "$PEM_PATH" "ubuntu@$ip" \
    "sudo tail -f /var/log/user-data.log"
}

# =============================================================================
# Subcommand: monitor — poll every N seconds, print latest progress
# =============================================================================
cmdMonitor() {
  requireTfState
  local id; id="$(instanceId)"
  local intervalSec="${1:-300}"   # default 300 sec (5 min)

  if [ -z "$id" ]; then
    echo "no instance in current state"

    return 1
  fi

  echo "Polling instance=$id every ${intervalSec}s. Ctrl+C to stop."

  while true; do
    local ts; ts="$(date '+%H:%M:%S')"

    # Get current state — possible values: pending|running|shutting-down|terminated
    local state
    state="$(aws ec2 describe-instances --instance-ids "$id" --region "$REGION" \
      --query 'Reservations[0].Instances[0].State.Name' --output text 2>/dev/null \
      || echo unknown)"

    # If the EC2 is gone, stop polling — nothing left to watch
    if [[ "$state" == "terminated" || "$state" == "shutting-down" ]]; then
      echo "[$ts] state=$state — instance is gone. Run './run.sh wait' to sync + cleanup."

      return 0
    fi

    # Get current public IP (sometimes it changes after a stop/start)
    local ip
    ip="$(aws ec2 describe-instances --instance-ids "$id" --region "$REGION" \
      --query 'Reservations[0].Instances[0].PublicIpAddress' --output text 2>/dev/null)"

    if [[ -z "$ip" || "$ip" == "None" ]]; then
      echo "[$ts] state=$state  ip=pending"
      sleep "$intervalSec"
      continue
    fi

    # Pull the last line of the boot log + the last line of any active
    # training log. SSH with a short connect-timeout so a hung instance
    # doesn't lock us up.
    local bootLine trainLine
    bootLine="$(ssh -o StrictHostKeyChecking=no -o BatchMode=yes \
                    -o ConnectTimeout=10 -i "$PEM_PATH" "ubuntu@$ip" \
                    "sudo tail -1 /var/log/user-data.log 2>/dev/null" 2>/dev/null \
                || echo "ssh-fail")"
    trainLine="$(ssh -o StrictHostKeyChecking=no -o BatchMode=yes \
                     -o ConnectTimeout=10 -i "$PEM_PATH" "ubuntu@$ip" \
                     "ls -1t /home/ubuntu/repo/checkpoints/motion_ssm/*/training.log 2>/dev/null \
                      | head -1 | xargs -r tail -1" 2>/dev/null || echo "")"
    echo "[$ts] state=$state ip=$ip"
    echo "       boot:  ${bootLine:0:120}"

    if [[ -n "$trainLine" ]]; then
      echo "       train: ${trainLine:0:120}"
    fi
    sleep "$intervalSec"
  done
}

# =============================================================================
# Subcommand: wait — block until EC2 self-terminates, then sync + cleanup
# =============================================================================
cmdWait() {
  requireTfState
  local id; id="$(instanceId)"

  if [ -z "$id" ]; then
    echo "no instance to wait for"

    return 0
  fi

  echo "Waiting for $id to self-terminate (this is normal — startup.sh"
  echo "calls terminate-instances after training completes)..."

  # `aws ec2 wait` is a built-in poller that blocks until the condition
  # is met. Cheaper than us writing our own polling loop.
  aws ec2 wait instance-terminated --instance-ids "$id" --region "$REGION"
  echo "Instance terminated."

  # Pull final checkpoints down. Same `aws s3 sync` as in startup.sh,
  # just in the reverse direction.
  if [[ -n "$S3_BUCKET" ]]; then
    echo "Syncing checkpoints from S3 -> ../../../checkpoints/"
    mkdir -p ../../../checkpoints
    aws s3 sync "s3://$S3_BUCKET/checkpoints" ../../../checkpoints
  else
    echo "WARN: s3_bucket not set in tfvars — skipping sync."
  fi

  # The EC2 is gone, but the IAM role + security group are still there.
  # `terraform destroy` removes them so the AWS console stays tidy.
  echo "Destroying remaining IAM role + security group..."
  terraform destroy -auto-approve
  echo "All resources destroyed. Bill = the time the EC2 was running."
}

# =============================================================================
# Subcommand: sync — download checkpoints from S3 to local
# =============================================================================
cmdSync() {
  if [[ -z "$S3_BUCKET" ]]; then
    echo "ERROR: s3_bucket not set in terraform.tfvars" >&2

    return 1
  fi
  mkdir -p ../../../checkpoints
  aws s3 sync "s3://$S3_BUCKET/checkpoints" ../../../checkpoints
}

# =============================================================================
# Subcommand: down — emergency: force-delete everything
# =============================================================================
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
  up         Provision EC2 spot + start training (asks confirm).
  status     Show current EC2 state.
  logs       SSH and `tail -f` /var/log/user-data.log on the EC2.
  monitor    Poll every 5 min: state + last boot/training log line.
             Pass seconds to override interval (e.g. ./run.sh monitor 60).
  wait       Block until EC2 self-terminates, then sync + cleanup.
  sync       Download checkpoints from S3 to ./checkpoints.
  down       Emergency: force-destroy everything (asks confirm).
  help       This message.

Typical happy path (read CLOUD-GUIDE.md first):
  ./run.sh up
  ./run.sh monitor     # walk away, check periodically
  ./run.sh wait        # auto-sync + destroy when training is done
EOF
}

# =============================================================================
# Dispatch: route the first argument to the right cmdX function
# =============================================================================
case "${1:-help}" in
  up)      cmdUp ;;
  status)  cmdStatus ;;
  logs)    cmdLogs ;;
  monitor) shift; cmdMonitor "$@" ;;
  wait)    cmdWait ;;
  sync)    cmdSync ;;
  down)    cmdDown ;;
  help|-h|--help) cmdHelp ;;
  *) echo "unknown subcommand: $1" >&2; cmdHelp; exit 1 ;;
esac
