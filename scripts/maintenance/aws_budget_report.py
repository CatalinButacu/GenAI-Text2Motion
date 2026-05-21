"""Print AWS cost summary for the last 90 days.

Read-only -- reads from Cost Explorer (`aws ce`). No side effects.

Run:
    python scripts/maintenance/aws_budget_report.py
"""

from __future__ import annotations

import datetime as dt
import json
import subprocess


def aws_ce(start: str, end: str, granularity: str = "MONTHLY") -> dict:
    cmd = [
        "aws", "ce", "get-cost-and-usage",
        "--time-period", f"Start={start},End={end}",
        "--granularity", granularity,
        "--metrics", "BlendedCost",
        "--group-by", "Type=DIMENSION,Key=SERVICE",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return json.loads(proc.stdout)


def main() -> None:
    today = dt.date.today()
    start = (today - dt.timedelta(days=90)).isoformat()
    end = today.isoformat()

    print(f"AWS BlendedCost {start} -> {end}\n")

    data = aws_ce(start, end, granularity="MONTHLY")
    grand_total = 0.0

    for period in data["ResultsByTime"]:
        ps, pe = period["TimePeriod"]["Start"], period["TimePeriod"]["End"]
        # Total is empty when there are only Groups -- sum from groups in that case.
        total_block = period.get("Total") or {}
        if "BlendedCost" in total_block:
            total = float(total_block["BlendedCost"]["Amount"])
        else:
            total = sum(
                float(g["Metrics"]["BlendedCost"]["Amount"])
                for g in period["Groups"]
            )
        grand_total += total
        print(f"  {ps} -> {pe}   ${total:8.2f}")
        for group in period["Groups"]:
            amt = float(group["Metrics"]["BlendedCost"]["Amount"])

            if amt < 0.01:
                continue
            svc = group["Keys"][0]
            print(f"      {svc:<40}  ${amt:7.2f}")

    print(f"\n  GRAND TOTAL (90d):                            ${grand_total:8.2f}")


if __name__ == "__main__":
    main()
