from __future__ import annotations


def log_action(name: str, detail: str = "") -> None:
    print(f"[UI] {name}{f' {detail}' if detail else ''}", flush=True)


def log_toggle(name: str, value: bool) -> None:
    print(f"[UI] {name} -> {'ON' if value else 'OFF'}", flush=True)


def log_result(name: str, ok: bool, detail: str = "") -> None:
    print(
        f"[UI] {name} -> {'OK' if ok else 'FAILED'}{f' ({detail})' if detail else ''}", flush=True
    )
