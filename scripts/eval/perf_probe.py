from __future__ import annotations

import argparse
import queue
import statistics
import threading
import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import replace

import torch

from text2motion.app.config import load_config
from text2motion.app.runtime import resolve_device, seed_everything
from text2motion.generation.model import Backbone, GeneratorArchitecture
from text2motion.generation.trainer import GeneratorTrainer
from text2motion.tokenization.model import build_tokenizer_module

FRAMES = 196


def build_trainer(config, device: str, backbone: Backbone, tokenizer_ckpt: str):
    tokenizer = build_tokenizer_module(config.tokenizer, config.rvq_baseline)
    tokenizer.load_state_dict(torch.load(tokenizer_ckpt, map_location="cpu"))
    tokenizer.to(device).eval().requires_grad_(False)
    architecture = GeneratorArchitecture.resolve(
        config.generator, backbone, tokenizer.codebook_size, config.tokenizer.num_quantizers
    )
    generator = architecture.build(device)
    return GeneratorTrainer(
        generator,
        tokenizer,
        config.train,
        downsample=config.tokenizer.downsample,
    )


def fixed_batch(batch: int, device: str, d_text: int):
    generator = torch.Generator(device="cpu").manual_seed(1234)
    motion = torch.randn(batch, FRAMES, 263, generator=generator).to(device)
    text = torch.randn(batch, 1, d_text, generator=generator).to(device)
    lengths = torch.full((batch,), FRAMES, device=device)
    return motion, text, lengths


def run_steps(config, device: str, backbone: Backbone, tokenizer_ckpt: str, steps: int) -> dict:
    seed_everything(config.seed, config.deterministic)
    trainer = build_trainer(config, device, backbone, tokenizer_ckpt)
    motion, text, lengths = fixed_batch(2, device, config.generator.d_text)
    for _ in range(steps):
        trainer.train_step(motion, text, lengths)
    return {name: value.detach().clone() for name, value in trainer.generator.state_dict().items()}


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


def median_ms(fn: Callable[[], None], repeats: int) -> float:
    fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1000)
    return statistics.median(samples)


def step_benchmark(args) -> None:
    config = load_config(args.config)
    device = resolve_device(config.device, args.device)
    seed_everything(config.seed, config.deterministic)
    trainer = build_trainer(config, device, args.backbone, args.tokenizer_ckpt)
    motion, text, lengths = fixed_batch(args.batch_size, device, config.generator.d_text)

    torch.cuda.reset_peak_memory_stats()
    ms = median_ms(lambda: trainer.train_step(motion, text, lengths), args.repeats)
    peak = torch.cuda.max_memory_allocated() / 1e9
    print(
        f"{args.backbone.value:<12} batch {args.batch_size:<3} "
        f"{ms:8.1f} ms/step  {ms / args.batch_size:7.1f} ms/clip  peak {peak:5.2f} GB"
    )


def sync_scan(args) -> None:
    config = load_config(args.config)
    device = resolve_device(config.device, args.device)
    seed_everything(config.seed, config.deterministic)
    trainer = build_trainer(config, device, args.backbone, args.tokenizer_ckpt)
    motion, text, lengths = fixed_batch(2, device, config.generator.d_text)

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
    tokenizer = build_tokenizer_module(config.tokenizer, config.rvq_baseline)
    tokenizer.load_state_dict(torch.load(args.tokenizer_ckpt, map_location="cpu"))
    architecture = GeneratorArchitecture.resolve(
        config.generator, args.backbone, tokenizer.codebook_size, config.tokenizer.num_quantizers
    )
    generator = architecture.build(device).eval()

    print(f"{'batch':>6} {'wall':>10} {'per clip':>10}")
    for batch in args.batches:
        text = torch.randn(
            batch, config.generator.text_prefix_len, config.generator.d_text, device=device
        )

        def rollout(text=text):
            with torch.no_grad():
                list(generator.stream(text, args.steps, temperature=1.0, top_p=0.9))

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
    from text2motion.app.container import ApplicationFactory
    from text2motion.app.schema import load_dataclass
    from text2motion.motion.representation import recover_skeleton
    from text2motion.streaming.decoder import StreamingMotionDecoder, measure_decoder_context
    from text2motion.studio.avatar import build_smplx_model, fit_smplx_to_joints
    from text2motion.studio.config import StudioConfig

    app = ApplicationFactory.from_config(args.config, device=args.device)
    tokenizer = app.tokenizer(args.tokenizer_ckpt)
    generator = app.generator(
        args.backbone,
        args.ckpt,
        tokenizer,
        tokenizer_checkpoint=args.tokenizer_ckpt,
        use_kernel=False,
    )
    device = app.device
    downsample = app.config.tokenizer.downsample
    studio = load_dataclass(StudioConfig, args.studio_config)
    scaler = app.motions().scaler()
    left, right = measure_decoder_context(tokenizer.module, downsample)
    decoder = StreamingMotionDecoder(
        tokenizer.module,
        downsample,
        torch.from_numpy(scaler.mean).to(device),
        torch.from_numpy(scaler.std).to(device),
        chunk_tokens=args.chunk_tokens,
        left_context=left,
        lookahead=right,
    )

    fit_base = replace(studio.fit, model_dir=args.model_dir)
    schedule = studio.schedule
    model_cache: dict[int, object] = {}

    def model_for(frames: int):
        if frames not in model_cache:
            model_cache[frames] = build_smplx_model(fit_base, frames, device)
        return model_cache[frames]

    if args.fit != "none":
        model_for(args.chunk_tokens * downsample)

    timer = StageTimer()
    with timer.stage("clip_encode"):
        with torch.no_grad():
            text_emb = generator.text_encoder([args.prompt])

    fit_queue: queue.Queue | None = None
    fit_state: dict = {"frames": 0, "err": []}

    def run_fit(joints, warm):
        config = replace(
            fit_base,
            stage1_iters=schedule.first_stage1_iters
            if warm is None
            else schedule.continuation_stage1_iters,
            stage2_iters=schedule.first_stage2_iters
            if warm is None
            else schedule.continuation_stage2_iters,
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
        fit_queue = queue.Queue()
        worker = threading.Thread(target=fit_consumer, daemon=True)
        worker.start()

    frames = 0
    warm_start = None
    token_iter = generator.module.stream(
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
    tokenizer = build_tokenizer_module(config.tokenizer, config.rvq_baseline)
    tokenizer.to(device).eval().requires_grad_(False)
    architecture = GeneratorArchitecture.resolve(
        config.generator,
        backbone,
        tokenizer.codebook_size,
        config.tokenizer.num_quantizers,
        use_kernel=use_kernel,
    )
    trainer = GeneratorTrainer(
        architecture.build(device), tokenizer, config.train, downsample=config.tokenizer.downsample
    )
    motion, text, lengths = fixed_batch(batch, device, config.generator.d_text)
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
            tokenizer = build_tokenizer_module(config.tokenizer, config.rvq_baseline)
            architecture = GeneratorArchitecture.resolve(
                config.generator,
                Backbone.MAMBA,
                tokenizer.codebook_size,
                config.tokenizer.num_quantizers,
                use_kernel=use_kernel,
            )
            generator = architecture.build(device).eval()
            text = torch.randn(
                1, config.generator.text_prefix_len, config.generator.d_text, device=device
            )

            def once(g=generator, t=text):
                with torch.no_grad():
                    list(g.stream(t, args.steps, temperature=1.0, top_p=0.9))

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
    parser.add_argument("--tokenizer_ckpt", default="checkpoints/tokenizer/tokenizer_fsq.pt")
    parser.add_argument("--backbone", default=Backbone.MAMBA, type=Backbone, choices=list(Backbone))
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--steps", type=int, default=49)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 8, 32])
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--studio_config", default="configs/studio.yml")
    parser.add_argument("--model_dir", default="data/smplx_models")
    parser.add_argument("--prompt", default="a person walks forward and waves")
    parser.add_argument("--cfg_scale", type=float, default=6.0)
    parser.add_argument("--chunk_tokens", type=int, default=4)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--fit", default="threaded", choices=("none", "serial", "threaded"))
    parser.add_argument("--rollout_only", action="store_true")
    parser.add_argument(
        "--probe",
        default="step",
        choices=("determinism", "step", "sync", "rollout", "demo", "kernel"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    {
        "determinism": determinism_check,
        "step": step_benchmark,
        "sync": sync_scan,
        "rollout": rollout_benchmark,
        "demo": demo_probe,
        "kernel": kernel_probe,
    }[args.probe](args)


if __name__ == "__main__":
    main()
