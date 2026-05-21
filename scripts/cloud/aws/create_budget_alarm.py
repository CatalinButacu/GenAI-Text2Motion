"""Create a hard $20 monthly AWS budget with email alerts at 50% and 100%.

Defensive measure for the dissertation training runs. The point isn't to cap
spend automatically -- AWS budgets are alert-only, they don't stop services --
but to make sure a forgotten EC2 instance can't run silently for days.

Run once:
    python scripts/cloud/aws/create_budget_alarm.py --email you@example.com

Cost: free. AWS budgets and SNS notifications under 1k/month are no-charge.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import date


def get_account_id() -> str:
    proc = subprocess.run(
        ["aws", "sts", "get-caller-identity", "--query", "Account", "--output", "text"],
        capture_output=True, text=True, check=True,
    )
    return proc.stdout.strip()


def budget_exists(account_id: str, name: str) -> bool:
    proc = subprocess.run(
        ["aws", "budgets", "describe-budgets", "--account-id", account_id],
        capture_output=True, text=True, check=False,
    )
    if proc.returncode != 0 or not proc.stdout.strip():
        return False
    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return False
    return any(b.get("BudgetName") == name for b in data.get("Budgets", []))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--email", required=True,
                        help="Email to alert when budget thresholds are crossed.")
    parser.add_argument("--limit", type=float, default=20.0,
                        help="Monthly USD cap (default $20).")
    parser.add_argument("--name", default="dissertation-training",
                        help="Budget name. Idempotent: skips creation if it already exists.")
    args = parser.parse_args()

    account_id = get_account_id()
    print(f"AWS account: {account_id}")

    if budget_exists(account_id, args.name):
        print(f"Budget {args.name!r} already exists. Nothing to do.")
        return 0

    today = date.today()
    # Budgets start at the first of a month.
    start = today.replace(day=1).isoformat() + "T00:00:00Z"

    budget = {
        "BudgetName": args.name,
        "BudgetLimit": {"Amount": f"{args.limit:.2f}", "Unit": "USD"},
        "TimeUnit": "MONTHLY",
        "TimePeriod": {"Start": start, "End": "2087-06-15T00:00:00Z"},
        "BudgetType": "COST",
        "CostTypes": {
            "IncludeTax": True,
            "IncludeSubscription": True,
            "UseBlended": False,
            "IncludeRefund": False,
            "IncludeCredit": False,
            "IncludeUpfront": True,
            "IncludeRecurring": True,
            "IncludeOtherSubscription": True,
            "IncludeSupport": True,
            "IncludeDiscount": True,
            "UseAmortized": False,
        },
    }

    notifications = [
        # 50% actual -- "heads up, run is still going"
        {
            "Notification": {
                "NotificationType": "ACTUAL",
                "ComparisonOperator": "GREATER_THAN",
                "Threshold": 50.0,
                "ThresholdType": "PERCENTAGE",
                "NotificationState": "ALARM",
            },
            "Subscribers": [{"SubscriptionType": "EMAIL", "Address": args.email}],
        },
        # 90% actual -- "wrap up"
        {
            "Notification": {
                "NotificationType": "ACTUAL",
                "ComparisonOperator": "GREATER_THAN",
                "Threshold": 90.0,
                "ThresholdType": "PERCENTAGE",
                "NotificationState": "ALARM",
            },
            "Subscribers": [{"SubscriptionType": "EMAIL", "Address": args.email}],
        },
        # 100% forecasted -- "kill it"
        {
            "Notification": {
                "NotificationType": "FORECASTED",
                "ComparisonOperator": "GREATER_THAN",
                "Threshold": 100.0,
                "ThresholdType": "PERCENTAGE",
                "NotificationState": "ALARM",
            },
            "Subscribers": [{"SubscriptionType": "EMAIL", "Address": args.email}],
        },
    ]

    cmd = [
        "aws", "budgets", "create-budget",
        "--account-id", account_id,
        "--budget", json.dumps(budget),
        "--notifications-with-subscribers", json.dumps(notifications),
    ]
    print(f"Creating ${args.limit:.0f}/month budget {args.name!r} -> {args.email}")
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)

    if proc.returncode != 0:
        print(f"  ERROR: {proc.stderr.strip()}", file=sys.stderr)
        return proc.returncode

    print("  OK. Check email -- AWS sends a one-time confirmation.")
    print("  View in console: https://console.aws.amazon.com/billing/home#/budgets")
    return 0


if __name__ == "__main__":
    sys.exit(main())
