# python scripts/benchmarks/perf_probe.py --probe demo --model_dir data/smplx
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import queue
import statistics
import threading
import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from text2motion.app.bootstrap import apply_precision, resolve_device, seed_everything
from text2motion.app.config import Float32Precision, load_config
from text2motion.generation.contracts import Backbone, MixedPrecisionMode
from text2motion.generation.nn_model import GeneratorModelSpec
from text2motion.generation.trainer import GeneratorTrainer
from text2motion.motion.representation import DIM, FPS
from text2motion.motion.storage import atomic_numpy_save, sha256_file
from text2motion.tokenization.nn_model import build_tokenizer_network


def build_trainer(config, device: str, backbone: Backbone, tokenizer_ckpt: str):
    tokenizer = build_tokenizer_network(config.tokenizer, config.rvq_baseline)
    tokenizer.load_state_dict(torch.load(tokenizer_ckpt, map_location="cpu"))
    tokenizer.to(device).eval().requires_grad_(False)
    architecture = GeneratorModelSpec.resolve(
        config.generator, backbone, tokenizer.codebook_size, config.tokenizer.num_quantizers
    )
    generator = architecture.create_model(device)
    return GeneratorTrainer(
        generator,
        tokenizer,
        config.train,
        downsample=config.tokenizer.downsample,
    )


def fixed_batch(batch: int, frames: int, device: str, d_text: int):
    generator = torch.Generator(device="cpu").manual_seed(2026)
    motion = torch.randn(batch, frames, DIM, generator=generator).to(device)
    text = torch.randn(batch, 1, d_text, generator=generator).to(device)
    lengths = torch.full((batch,), frames, device=device)
    return motion, text, lengths


def run_steps(config, device: str, backbone: Backbone, tokenizer_ckpt: str, steps: int) -> dict:
    seed_everything(config.seed, config.deterministic)
    trainer = build_trainer(config, device, backbone, tokenizer_ckpt)
    motion, text, lengths = fixed_batch(
        2, config.data.max_motion_len, device, config.generator.d_text
    )
    for _ in range(steps):
        trainer.train_step(motion, text, lengths)
    return {
        name: value.detach().clone() for name, value in trainer.token_generator.state_dict().items()
    }


def determinism_check(args) -> None:
    config = load_config(args.config)
    device = resolve_device(config.device, args.device)
    first = run_steps(config, device, args.backbone, args.tokenizer_ckpt, args.steps)
    second = run_steps(config, device, args.backbone, args.tokenizer_ckpt, args.steps)

    mismatched = [name for name in first if not torch.equal(first[name], second[name])]
    worst = 0.0
    for name in mismatched:
        worst = max(worst, float((first[name] - second[name]).abs().max()))

    print(f"device {device}  backbone {args.backbone.value}  steps {args.steps}")
    print(f"deterministic={config.deterministic}  CUBLAS_WORKSPACE_CONFIG is set at import")
    print(f"tensors compared: {len(first)}   bit-identical: {len(first) - len(mismatched)}")
    if mismatched:
        print(f"NOT bit-exact: {len(mismatched)} tensors differ, max |diff| {worst:.3e}")
        print("  first offenders:", mismatched[:5])
        print("  -> the paper must say 'reproducible within' this bound, not 'bit-for-bit'")
    else:
        print("BIT-EXACT across two fresh runs -- the determinism claim holds")


def median_ms(
    fn: Callable[[], object], repeats: int, durations: list[float] | None = None
) -> float:
    fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1000)
    if durations is not None:
        durations.extend(samples)
    return statistics.median(samples)


def precision_error(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    if reference.shape != candidate.shape or not reference.size:
        raise ValueError("precision comparison requires matching nonempty arrays")
    if not np.isfinite(reference).all() or not np.isfinite(candidate).all():
        raise ValueError("precision comparison requires finite arrays")
    base = reference.astype(np.float64).ravel()
    delta = candidate.astype(np.float64).ravel() - base
    norm, error = float(np.linalg.norm(base)), float(np.linalg.norm(delta))
    if norm == 0 and error != 0:
        raise ValueError("nonzero candidate has undefined relative error against a zero reference")
    return {
        "max_abs": float(np.abs(delta).max()),
        "relative_l2": error / norm if norm else 0.0,
    }


def precision_numeric(trainer, text, tokens, amp: bool):
    model = trainer.token_generator
    model.eval()
    model.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
        logits = model(tokens, text)
        loss = torch.nn.functional.cross_entropy(logits.flatten(0, 2), tokens.flatten())
    if not torch.isfinite(logits).all() or not torch.isfinite(loss):
        raise RuntimeError("precision probe produced nonfinite forward values")
    loss.backward()
    gradients = []
    names = []
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            if not torch.isfinite(parameter.grad).all():
                raise RuntimeError(f"precision probe produced nonfinite gradient: {name}")
            names.append(name)
            gradients.append(parameter.grad.detach().float().cpu().flatten())
    if not gradients:
        raise RuntimeError("precision probe produced no parameter gradients")
    model.train()
    return (
        logits.detach().float().cpu().numpy(),
        torch.cat(gradients).numpy(),
        float(loss.detach()),
        names,
    )


def precision_case(args, config, device: str, backbone: Backbone, mode: str, tokens):
    amp = mode == "bf16"
    seed_everything(2026, config.deterministic)
    trainer = build_trainer(
        replace(
            config,
            train=replace(
                config.train, amp=MixedPrecisionMode.BF16 if amp else MixedPrecisionMode.OFF
            ),
        ),
        device,
        backbone,
        args.tokenizer_ckpt,
    )
    digest = hashlib.sha256()
    for name, tensor in trainer.token_generator.state_dict().items():
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().numpy().tobytes())
    motion, text, lengths = fixed_batch(
        args.batch_size, args.frames[0], device, config.generator.d_text
    )
    apply_precision(Float32Precision.HIGHEST)
    if tokens is None:
        with torch.no_grad():
            tokens = trainer.tokenizer_network.encode(motion)
            if trainer.token_generator.end_id is not None:
                tokens, _ = trainer.append_end_targets(tokens, lengths)
    apply_precision(Float32Precision.HIGHEST if mode == "fp32" else Float32Precision.TF32)
    logits, gradients, loss, names = precision_numeric(trainer, text, tokens, amp)
    folder = args.out_path.with_suffix("")
    atomic_numpy_save(logits, folder / f"{backbone.value}_{mode}_logits.npy")
    atomic_numpy_save(gradients, folder / f"{backbone.value}_{mode}_gradients.npy")
    if mode == "fp32":
        atomic_numpy_save(tokens.cpu().numpy(), folder / f"{backbone.value}_tokens.npy")
    torch.cuda.reset_peak_memory_stats()
    durations = []
    ms = median_ms(
        lambda step=trainer.train_step: step(motion, text, lengths),
        args.repeats,
        durations=durations,
    )
    peak = torch.cuda.max_memory_allocated() / 1e9
    result = {
        "backbone": backbone.value,
        "mode": mode,
        "seed": 2026,
        "batch_size": args.batch_size,
        "frames": args.frames[0],
        "repeats": args.repeats,
        "ms_per_step": ms,
        "step_durations_ms": durations,
        "peak_allocated_gb": peak,
        "numeric_ce": loss,
        "initial_weights_sha256": digest.hexdigest(),
        "gradient_names": names,
        "grad_accum": config.train.grad_accum,
        "mamba_kernel": backbone == Backbone.MAMBA and config.generator.use_kernel,
    }
    del trainer
    torch.cuda.empty_cache()
    return result, logits, gradients, tokens


def precision_benchmark(args) -> None:
    if args.out_path is None or args.summary_csv is None:
        raise ValueError("precision probe requires --out_path and --summary_csv")
    if any(
        path.exists() for path in (args.out_path, args.summary_csv, args.out_path.with_suffix(""))
    ):
        raise FileExistsError("precision probe requires new output paths")
    config = load_config(args.config)
    device = resolve_device(config.device, args.device)
    if device != "cuda" or not torch.cuda.is_bf16_supported():
        raise RuntimeError("precision probe requires CUDA with BF16 support")
    if args.repeats < 1 or args.batch_size < 1 or len(args.frames) != 1:
        raise ValueError("precision probe requires positive repeats/batch and one frame length")
    if config.train.grad_accum != 1:
        raise ValueError(
            "precision step timing requires grad_accum1, one optimizer update per call"
        )
    report = {
        "status": "in_progress",
        "config": args.config,
        "config_sha256": sha256_file(args.config),
        "tokenizer_checkpoint": args.tokenizer_ckpt,
        "tokenizer_sha256": sha256_file(args.tokenizer_ckpt),
        "seed": 2026,
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "rows": [],
        "limitations": "Numeric forward/backward compares clean token-CE on fixed strict-FP32 tokenizer targets "
        "and identical initial generator weights with dropout off. It does not prove total-loss gradient fidelity. "
        "Timing uses the configured generator/tokenizer train_step with corruption/dropout, live tokenization, soft decode and optimizer/EMA; "
        "synthetic precomputed text excludes CLIP encoding/backprop and the data loader. "
        "Synthetic motion/text, one fixed length, no epoch-speed or generalization claim. BF16 autocast plus TF32 FP32 operations.",
    }
    csv_rows = []
    for backbone in Backbone:
        tokens, reference, reference_grads, weight_hash, reference_names = (
            None,
            None,
            None,
            None,
            None,
        )
        for mode in ("fp32", "tf32", "bf16"):
            row, logits, gradients, tokens = precision_case(
                args, config, device, backbone, mode, tokens
            )
            if reference is None or reference_grads is None:
                reference, reference_grads = logits, gradients
                weight_hash, reference_names = row["initial_weights_sha256"], row["gradient_names"]
            if (
                row["initial_weights_sha256"] != weight_hash
                or row["gradient_names"] != reference_names
            ):
                raise RuntimeError(
                    "precision modes changed initial weights or gradient parameter coverage"
                )
            forward = precision_error(reference, logits)
            backward = precision_error(reference_grads, gradients)
            row.update(
                logits_max_abs=forward["max_abs"],
                logits_relative_l2=forward["relative_l2"],
                gradients_max_abs=backward["max_abs"],
                gradients_relative_l2=backward["relative_l2"],
            )
            report["rows"].append(row)
            csv_rows.append(
                {k: v for k, v in row.items() if k not in ("gradient_names", "step_durations_ms")}
            )
            args.out_path.write_text(
                json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
            )
            print(
                f"K8 {backbone.value}/{mode}: {row['ms_per_step']:.1f} ms/step; peak {row['peak_allocated_gb']:.3f} GB; logits relL2 {row['logits_relative_l2']:.3g}; grads relL2 {row['gradients_relative_l2']:.3g}",
                flush=True,
            )
        del reference, reference_grads, tokens
    report["status"] = "complete"
    args.out_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.summary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)


def step_benchmark(args) -> None:
    config = load_config(args.config)
    device = resolve_device(config.device, args.device)
    seed_everything(config.seed, config.deterministic)
    trainer = build_trainer(config, device, args.backbone, args.tokenizer_ckpt)
    for frames in args.frames:
        motion, text, lengths = fixed_batch(
            args.batch_size, frames, device, config.generator.d_text
        )
        torch.cuda.reset_peak_memory_stats()
        ms = median_ms(
            lambda m=motion, t=text, n=lengths: trainer.train_step(m, t, n), args.repeats
        )
        peak = torch.cuda.max_memory_allocated() / 1e9
        print(
            f"{args.backbone.value:<12} batch {args.batch_size:<3} frames {frames:<4} "
            f"{ms:8.1f} ms/step  {ms / args.batch_size:7.1f} ms/clip  peak {peak:5.2f} GB"
        )


def profile_step(args) -> None:
    config = load_config(args.config)
    device = resolve_device(config.device, args.device)
    seed_everything(config.seed, config.deterministic)
    trainer = build_trainer(config, device, args.backbone, args.tokenizer_ckpt)
    frames = args.frames[0]
    motion, text, lengths = fixed_batch(args.batch_size, frames, device, config.generator.d_text)
    for _ in range(2):
        trainer.train_step(motion, text, lengths)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    with torch.profiler.profile(activities=activities, profile_memory=True) as profiler:
        trainer.train_step(motion, text, lengths)
        torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() / 1e9
    averages = profiler.key_averages()
    print(averages.table(sort_by="self_device_time_total", row_limit=args.rows))
    print(f"{args.backbone.value} batch {args.batch_size} frames {frames} peak {peak:.2f} GB")


def sync_scan(args) -> None:
    config = load_config(args.config)
    device = resolve_device(config.device, args.device)
    seed_everything(config.seed, config.deterministic)
    trainer = build_trainer(config, device, args.backbone, args.tokenizer_ckpt)
    motion, text, lengths = fixed_batch(
        2, config.data.max_motion_len, device, config.generator.d_text
    )

    trainer.train_step(motion, text, lengths)
    torch.cuda.set_sync_debug_mode("warn")
    try:
        trainer.train_step(motion, text, lengths)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    print("sync scan complete -- every warning above is one GPU->CPU stall inside train_step")


def rollout_benchmark(args) -> None:
    config = load_config(args.config)
    device = resolve_device(config.device, args.device)
    seed_everything(config.seed, config.deterministic)
    tokenizer = build_tokenizer_network(config.tokenizer, config.rvq_baseline)
    tokenizer.load_state_dict(torch.load(args.tokenizer_ckpt, map_location="cpu"))
    architecture = GeneratorModelSpec.resolve(
        config.generator, args.backbone, tokenizer.codebook_size, config.tokenizer.num_quantizers
    )
    generator = architecture.create_model(device).eval()

    print(f"{'batch':>6} {'wall':>10} {'per clip':>10}")
    for batch in args.batches:
        text = torch.randn(
            batch, config.generator.text_prefix_len, config.generator.d_text, device=device
        )

        def rollout(text=text):
            with torch.no_grad():
                list(generator.stream_token_indices(text, args.steps, temperature=1.0, top_p=0.9))

        ms = median_ms(rollout, args.repeats)
        print(f"{batch:>6} {ms / 1000:>9.3f}s {ms / batch:>9.1f}ms")


class StageTimer:
    def __init__(self) -> None:
        self.totals: dict[str, float] = {}
        self.origin = time.perf_counter()
        self.marks: dict[str, float] = {}

    @contextmanager
    def stage(self, name: str):
        torch.cuda.synchronize()
        start = time.perf_counter()
        try:
            yield
        finally:
            torch.cuda.synchronize()
            self.totals[name] = self.totals.get(name, 0.0) + (time.perf_counter() - start)

    def mark(self, name: str) -> None:
        self.marks.setdefault(name, time.perf_counter() - self.origin)

    def report(self, label: str, frames: int, fps: int) -> None:
        wall = time.perf_counter() - self.origin
        print(f"\n=== {label} ===")
        for name, seconds in self.totals.items():
            print(f"  {name:<26} {seconds * 1000:9.1f} ms")
        print(f"  {'-' * 38}")
        for name, when in sorted(self.marks.items(), key=lambda kv: kv[1]):
            print(f"  {name:<26} {when * 1000:9.1f} ms  (from request)")
        motion = frames / fps
        print(f"  {'TOTAL WALL':<26} {wall * 1000:9.1f} ms")
        print(f"  {'motion produced':<26} {motion:9.1f} s  -> {motion / wall:5.1f}x realtime")


def demo_probe(args) -> None:
    if args.model_dir is None:
        raise SystemExit(
            "the demo probe fits SMPL-X bodies: pass --model_dir or set paths.smplx_models"
        )
    from text2motion.app.bootstrap import ApplicationBootstrap
    from text2motion.app.config import load_dataclass
    from text2motion.generation.decoder import StreamingMotionDecoder, measure_decoder_context
    from text2motion.motion.representation import recover_skeleton
    from text2motion.studio.config import StudioConfig
    from text2motion.studio.fitting import build_smplx_body_model, fit_smplx_to_joints

    app = ApplicationBootstrap.from_config(args.config, device=args.device)
    tokenizer = app.load_tokenizer(args.tokenizer_ckpt)
    generator = app.load_text_to_motion_generator(
        args.backbone,
        args.ckpt,
        tokenizer,
        tokenizer_checkpoint=args.tokenizer_ckpt,
        use_kernel=False,
    )
    device = app.device
    downsample = app.config.tokenizer.downsample
    studio = load_dataclass(StudioConfig, args.studio_config)
    scaler = app.create_motion_repository().load_scaler()
    left, right = measure_decoder_context(tokenizer.network, downsample)
    decoder = StreamingMotionDecoder(
        tokenizer.network,
        downsample,
        torch.from_numpy(scaler.mean).to(device),
        torch.from_numpy(scaler.std).to(device),
        chunk_tokens=args.chunk_tokens,
        left_context=left,
        lookahead=right,
    )

    fit_base = replace(studio.fit, model_dir=args.model_dir)
    model_cache: dict[int, object] = {}

    def model_for(frames: int):
        if frames not in model_cache:
            model_cache[frames] = build_smplx_body_model(fit_base, frames, device)
        return model_cache[frames]

    if args.fit != "none":
        print("body fit: offline iterative fitter (the live studio uses the analytic solver)")
        model_for(args.chunk_tokens * downsample)

    timer = StageTimer()
    with timer.stage("clip_encode"), torch.no_grad():
        text_emb = generator.text_encoder([args.prompt])

    fit_queue: queue.Queue = queue.Queue()
    fit_state: dict = {"frames": 0, "err": []}

    def run_fit(joints, warm):
        config = replace(
            fit_base,
            stage1_iters=args.first_stage1_iters if warm is None else args.next_stage1_iters,
            stage2_iters=args.first_stage2_iters if warm is None else args.next_stage2_iters,
        )
        result = fit_smplx_to_joints(
            joints, config, device=device, model=model_for(len(joints)), warm_start=warm
        )
        fit_state["frames"] += len(joints)
        fit_state["err"].append(result.joint_err_cm)
        timer.mark("first body chunk")
        return (result.global_orient[-1], result.body_pose[-1], result.betas)

    def fit_consumer():
        warm = None
        while True:
            item = fit_queue.get()
            if item is None:
                break
            start = time.perf_counter()
            warm = run_fit(item, warm)
            timer.totals["fit (threaded)"] = timer.totals.get("fit (threaded)", 0.0) + (
                time.perf_counter() - start
            )

    worker = None
    if args.fit == "threaded":
        worker = threading.Thread(target=fit_consumer, daemon=True)
        worker.start()

    frames = 0
    warm_start = None
    token_iter = generator.token_generator.stream_token_indices(
        text_emb, args.steps, temperature=1.0, top_p=0.9, cfg_scale=args.cfg_scale
    )

    generated = time.perf_counter()
    for chunk in decoder.stream_tokens(token_iter):
        timer.totals["generate + decode"] = timer.totals.get("generate + decode", 0.0) + (
            time.perf_counter() - generated
        )
        with timer.stage("recover_skeleton"):
            joints = recover_skeleton(chunk.squeeze(0).cpu().numpy())
        frames += len(joints)
        timer.mark("first skeleton chunk")
        if args.fit == "threaded":
            fit_queue.put(joints)
        elif args.fit == "serial":
            with timer.stage("fit (inline, blocks stream)"):
                warm_start = run_fit(joints, warm_start)
        generated = time.perf_counter()

    timer.mark("generation complete")
    if worker is not None:
        fit_queue.put(None)
        worker.join()
    timer.mark("body complete")

    label = f"DEMO {args.backbone.value} steps={args.steps} cfg={args.cfg_scale} fit={args.fit}"
    timer.report(label, frames, args.fps)
    if fit_state["err"]:
        mean_err = sum(fit_state["err"]) / len(fit_state["err"])
        print(f"  {'fit chunks':<26} {len(fit_state['err']):9d}   mean joint err {mean_err:.2f} cm")


def build_kernel_trainer(config, device, backbone, use_kernel: bool, batch: int):
    tokenizer = build_tokenizer_network(config.tokenizer, config.rvq_baseline)
    tokenizer.to(device).eval().requires_grad_(False)
    architecture = GeneratorModelSpec.resolve(
        config.generator,
        backbone,
        tokenizer.codebook_size,
        config.tokenizer.num_quantizers,
        use_kernel=use_kernel,
    )
    trainer = GeneratorTrainer(
        architecture.create_model(device),
        tokenizer,
        config.train,
        downsample=config.tokenizer.downsample,
    )
    motion, text, lengths = fixed_batch(
        batch, config.data.max_motion_len, device, config.generator.d_text
    )
    return trainer, (motion, text, lengths)


def kernel_probe(args) -> None:
    import importlib.util

    if importlib.util.find_spec("mamba_ssm") is None:
        raise SystemExit("mamba_ssm is not installed -- gate 1 must pass before this probe")

    config = load_config(args.config)
    device = resolve_device(config.device, args.device)
    seed_everything(config.seed, config.deterministic)

    print(f"config {args.config}  device {device}  batch {args.batch_size}")
    print("note: checkpoint_blocks = not use_kernel, so the fused run also drops gradient")
    print(
        "      checkpointing -- that is the real deployed difference, not an isolated kernel swap"
    )

    if not args.rollout_only:
        results = {}
        for _ in range(2):
            for use_kernel in (False, True):
                trainer, batch = build_kernel_trainer(
                    config, device, Backbone.MAMBA, use_kernel, args.batch_size
                )
                torch.cuda.reset_peak_memory_stats()
                ms = median_ms(lambda t=trainer, b=batch: t.train_step(*b), args.repeats)
                peak = torch.cuda.max_memory_allocated() / 1e9
                key = "fused" if use_kernel else "eager"
                if key not in results or ms < results[key][0]:
                    results[key] = (ms, peak)
                del trainer
                torch.cuda.empty_cache()

        eager_ms, eager_peak = results["eager"]
        fused_ms, fused_peak = results["fused"]
        print()
        print(f"{'mamba train step':<22} {'ms':>10} {'peak GB':>10}")
        print(f"{'  eager scan':<22} {eager_ms:>9.1f} {eager_peak:>10.2f}")
        print(f"{'  fused kernel':<22} {fused_ms:>9.1f} {fused_peak:>10.2f}")
        print(f"{'  speedup':<22} {eager_ms / fused_ms:>9.2f}x")

    rollout = {}
    for _ in range(2):
        for use_kernel in (False, True):
            tokenizer = build_tokenizer_network(config.tokenizer, config.rvq_baseline)
            architecture = GeneratorModelSpec.resolve(
                config.generator,
                Backbone.MAMBA,
                tokenizer.codebook_size,
                config.tokenizer.num_quantizers,
                use_kernel=use_kernel,
            )
            generator = architecture.create_model(device).eval()
            text = torch.randn(
                1, config.generator.text_prefix_len, config.generator.d_text, device=device
            )

            def once(g=generator, t=text):
                with torch.no_grad():
                    list(g.stream_token_indices(t, args.steps, temperature=1.0, top_p=0.9))

            ms = median_ms(once, max(3, args.repeats // 2))
            key = "fused" if use_kernel else "eager"
            if key not in rollout or ms < rollout[key]:
                rollout[key] = ms
            del generator
            torch.cuda.empty_cache()

    print()
    print(f"{'mamba rollout (' + str(args.steps) + ' tokens)':<30} {'ms':>10}")
    print(f"{'  eager scan':<30} {rollout['eager']:>9.1f}")
    print(f"{'  fused kernel':<30} {rollout['fused']:>9.1f}")
    print(f"{'  ratio':<30} {rollout['eager'] / rollout['fused']:>9.2f}x")
    print()
    print("expect ~1.00x: use_kernel only affects MambaMixer.forward (training).")
    print("Streaming uses MambaMixer.step, which has no fused path -- fixing that")
    print("needs mamba_ssm.ops.triton.selective_state_update wired into step().")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Performance and determinism probes.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--tokenizer_ckpt")
    parser.add_argument("--backbone", default=Backbone.MAMBA, type=Backbone, choices=list(Backbone))
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--out_path", type=Path)
    parser.add_argument("--summary_csv", type=Path)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--frames", type=int, nargs="+")
    parser.add_argument("--rows", type=int, default=15)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 8, 32])
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--studio_config", default="configs/studio.yml")
    parser.add_argument("--model_dir")
    parser.add_argument("--first_stage1_iters", type=int, default=20)
    parser.add_argument("--first_stage2_iters", type=int, default=50)
    parser.add_argument("--next_stage1_iters", type=int, default=0)
    parser.add_argument("--next_stage2_iters", type=int, default=25)
    parser.add_argument("--prompt", default="a person walks forward and waves")
    parser.add_argument("--cfg_scale", type=float, default=6.0)
    parser.add_argument("--chunk_tokens", type=int, default=4)
    parser.add_argument("--fps", type=int)
    parser.add_argument("--fit", default="threaded", choices=("none", "serial", "threaded"))
    parser.add_argument("--rollout_only", action="store_true")
    parser.add_argument(
        "--probe",
        default="step",
        choices=(
            "determinism",
            "step",
            "profile",
            "sync",
            "rollout",
            "demo",
            "kernel",
            "precision",
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    apply_precision(config.precision)
    args.tokenizer_ckpt = args.tokenizer_ckpt or str(config.paths.tokenizer_checkpoint)
    if args.probe == "demo":
        args.ckpt = args.ckpt or str(config.paths.generator_checkpoint(args.backbone))
    args.steps = args.steps or config.data.max_motion_len // config.tokenizer.downsample
    args.frames = args.frames or [config.data.max_motion_len]
    args.fps = args.fps or FPS
    args.model_dir = args.model_dir or (
        str(config.paths.smplx_models) if config.paths.smplx_models else None
    )
    {
        "determinism": determinism_check,
        "step": step_benchmark,
        "profile": profile_step,
        "sync": sync_scan,
        "rollout": rollout_benchmark,
        "demo": demo_probe,
        "kernel": kernel_probe,
        "precision": precision_benchmark,
    }[args.probe](args)


if __name__ == "__main__":
    main()
