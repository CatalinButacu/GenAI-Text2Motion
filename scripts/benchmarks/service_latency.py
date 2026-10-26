# python scripts/benchmarks/service_latency.py --config configs/generator/gen_pilot_fsq8x1024.yaml --backbone mamba --ckpt artifacts/checkpoints/generator/generator_mamba_bs8_e60.pt
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from text2motion.app.config import load_config, load_service_config, service_policy
from text2motion.streaming.protocol import motion_service_client

DELIVERY_SCOPE = (
    "delivery through the motion service: trained checkpoint, CLIP text encoder, sampling with "
    "guidance, tokenizer decoding and socket transport to the client; no body fit or display"
)
PROMPTS = (
    "a person walks forward",
    "a person jumps up and down",
    "a person waves with the right hand",
    "a person turns around and sits down",
    "a person runs in a circle",
)
SUMMARY_KEYS = (
    "first_chunk_ms",
    "ms_per_step",
    "total_ms",
    "text_encode_ms",
    "realtime_factor",
    "peak_gpu_gb",
    "host_rss_gb",
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Time generation through the motion service.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--backbone", required=True, choices=["mamba", "transformer"])
    parser.add_argument("--ckpt", required=True)
    parser.add_argument(
        "--tokenizer_ckpt", default="artifacts/checkpoints/tokenizer/fsq_g8_v1024.pt"
    )
    parser.add_argument("--service_config", default="configs/service.yml")
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--cfg_scale", type=float, default=6.0)
    parser.add_argument("--out", default="artifacts/outputs/figures/service_latency.json")
    return parser.parse_args()


def timed_requests(client, args: argparse.Namespace) -> list[dict]:
    rows = []
    for _ in range(args.repeats):
        for prompt in PROMPTS:
            for _chunk in client.generate(
                prompt,
                steps=args.steps,
                temperature=args.temperature,
                top_p=args.top_p,
                cfg_scale=args.cfg_scale,
            ):
                pass
            rows.append({"prompt": prompt, **client.last_stats})
    return rows


def main() -> None:
    args = parse_arguments()
    service = load_service_config(args.service_config)
    client = motion_service_client(
        args.config,
        args.ckpt,
        args.tokenizer_ckpt,
        args.backbone,
        service.connect_host,
        service.port,
        load_config(args.config).paths.service_log,
        service.startup_timeout_seconds,
        service_config=args.service_config,
        policy=service_policy(service),
    )
    rows = timed_requests(client, args)
    summary = {
        "scope": DELIVERY_SCOPE,
        "identity": client.hello["identity"],
        "policy": client.hello["policy"],
        "steps": args.steps,
        "requests": len(rows),
        "median": {key: statistics.median(row[key] for row in rows) for key in SUMMARY_KEYS},
        "rows": rows,
    }
    client.close()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    for key, value in summary["median"].items():
        print(f"{key:<18} {value:10.3f}")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
