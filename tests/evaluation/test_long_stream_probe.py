import json
import sys
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from scripts.qa import long_stream_probe as probe
from scripts.qa.long_stream_probe import MemoryPolicy, paired_change, policy_fits

from text2motion.motion.contracts import Split
from text2motion.motion.representation import RAW_OFFSETS, kinematic_bones


def test_review_selection_is_balanced_shared_and_repeatable():
    rows = [
        {
            "backbone": backbone,
            "offset": offset,
            "pair": pair,
            "root_step_delta_m": float(pair),
            "skate_delta_m_per_s": None,
            "early_margin_delta": -float(pair),
        }
        for backbone in ("mamba", "transformer")
        for offset in (0, 64)
        for pair in range(32)
    ]
    selected = probe.review_selection(rows)
    assert selected == probe.review_selection(rows)
    for offset in (0, 64):
        indices = []
        for backbone in ("mamba", "transformer"):
            group = {
                x["row"]["pair"]
                for x in selected
                if x["group"] == "random"
                and x["row"]["backbone"] == backbone
                and x["row"]["offset"] == offset
            }
            assert len(group) == 8
            indices.append(group)
        assert indices[0] == indices[1]
    keys = [(x["row"]["backbone"], x["row"]["offset"], x["row"]["pair"]) for x in selected]
    assert len(keys) == len(set(keys))


@pytest.mark.parametrize("scenario", probe.COUNTERFACTUAL_SCENARIOS)
def test_counterfactual_selection_preserves_original_pair_identity(scenario):
    selection = [probe.ProbePrompt(str(i), str(i), f"caption {i}", ()) for i in range(64)]
    actual = probe.scenario_selection(selection, scenario)
    left, right = selection[:32], selection[32:]
    assert (
        actual
        == {
            "A_to_A": left + left,
            "A_to_B": left + right,
            "B_to_A": right + left,
        }[scenario]
    )
    assert selection == left + right
    with pytest.raises(ValueError, match="original32-pair"):
        probe.scenario_selection(selection[:32], scenario)
    with pytest.raises(ValueError, match="original32-pair"):
        probe.scenario_selection(selection, "unknown")


def test_counterfactual_interactions_use_identical_ab_targets_for_same_caption_control(tmp_path):
    old = np.arange(32, dtype=float)[:, None] * 10
    new = old + 2
    cases = {"mamba": {}}
    for scenario in ("A_to_A", "A_to_B"):
        cases["mamba"][scenario] = {}
        for keep in ("16", "0"):
            folder = tmp_path / f"{scenario}_{keep}"
            folder.mkdir()
            np.save(
                folder / "text_embeddings.npy",
                np.stack((old, old if scenario == "A_to_A" else new)),
            )
            motion = old if keep == "16" else old + (0.1 if scenario == "A_to_A" else 2.5)
            np.save(folder / "semantic_embeddings.npy", np.broadcast_to(motion, (3, 2, 32, 1)))
            cases["mamba"][scenario][keep] = {"path": str(folder.with_suffix(".json"))}
    actual = probe.counterfactual_interactions(cases)["mamba"]
    for window in ("full", "early", "late"):
        for field in ("last_minus_first", "ci95_low", "ci95_high"):
            assert actual[window]["new_caption_margin"][field] == pytest.approx(3.8)
        assert actual[window]["new_caption_preferred"]["last_minus_first"] == 1


def test_counterfactual_cli_routes_shared_screen(monkeypatch):
    calls = []
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "probe",
            "--counterfactual",
            "--compare",
            "mamba.json",
            "transformer.json",
            "--family_index",
            "index.csv",
            "--out_path",
            "out.json",
            "--summary_csv",
            "out.csv",
        ],
    )
    monkeypatch.setattr(
        probe, "history_screen", lambda *args, **kwargs: calls.append((args, kwargs))
    )
    probe.main()
    assert calls[0][0] == (
        [Path("mamba.json"), Path("transformer.json")],
        Path("out.json"),
        Path("out.csv"),
        Path("index.csv"),
    )
    assert calls[0][1] == {"counterfactual": True}


def test_review_cli_accepts_one_archive_without_loading_models(monkeypatch):
    calls = []
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "probe",
            "--review_audit",
            "--compare",
            "source.json",
            "--out_path",
            "audit.json",
            "--summary_csv",
            "pairs.csv",
        ],
    )
    monkeypatch.setattr(probe, "review_audit", lambda *args: calls.append(args))
    probe.main()
    assert calls == [(Path("source.json"), Path("audit.json"), Path("pairs.csv"))]
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "probe",
            "--compare",
            "source.json",
            "--out_path",
            "audit.json",
            "--summary_csv",
            "pairs.csv",
        ],
    )
    with pytest.raises(SystemExit):
        probe.main()


def test_switch_scores_use_new_caption_retrieval_and_positive_old_minus_new_margin():
    old = np.arange(32, dtype=float)[:, None] * 10
    new = old + 2
    scores = probe.switch_metrics(np.stack((old, new)), new + 0.25)
    np.testing.assert_array_equal(scores["new_r_top1"], np.ones(32))
    np.testing.assert_allclose(scores["new_text_distance"], 0.25)
    np.testing.assert_allclose(scores["old_text_distance"], 2.25)
    np.testing.assert_allclose(scores["new_caption_margin"], 2)
    np.testing.assert_array_equal(scores["new_caption_preferred"], np.ones(32))
    tied = probe.switch_metrics(np.stack((new, new)), new)
    assert tied["new_caption_preferred"].sum() == 0
    assert tied["caption_distance_tie"].sum() == 32


@pytest.mark.parametrize("shape", [(2, 31, 4), (32, 4)])
def test_switch_scores_reject_incomplete_or_malformed_caption_pools(shape):
    with pytest.raises(ValueError, match="32-pair pool"):
        probe.switch_metrics(np.zeros(shape), np.zeros((32, 4)))


def test_switch_scores_reject_nonfinite_motion_embeddings():
    with pytest.raises(ValueError, match="finite"):
        probe.switch_metrics(np.zeros((2, 32, 4)), np.full((32, 4), np.nan))


def test_switch_contrasts_pair_methods_and_match_early_late_windows():
    measured = {
        "mamba": {
            "full": {"cut": {"score": np.ones(32)}, "generator": {"score": np.full(32, 2.0)}},
            "early": {"cut": {"score": np.ones(32)}, "generator": {"score": np.full(32, 2.0)}},
            "late": {"cut": {"score": np.full(32, 3.0)}, "generator": {"score": np.full(32, 6.0)}},
        }
    }
    contrasts = probe.switch_contrasts(measured)
    assert contrasts["mamba/late/generator_minus_cut"]["score"]["last_minus_first"] == 3
    assert contrasts["mamba/generator/late_minus_early"]["score"] == {
        "last_minus_first": 4,
        "ci95_low": 4,
        "ci95_high": 4,
    }


def test_history_contrasts_use_keep16_and_jointly_contacting_pairs(tmp_path):
    folder = tmp_path / "base"
    folder.mkdir()
    np.save(folder / "reference_jerk.npy", np.ones((32, 2, 57)))
    variants = {}
    for keep, level, skate in (("16", 1.0, 0.2), ("8", 0.5, 0.3), ("0", 0.25, None)):
        variants[keep] = {
            "path": str(folder.with_suffix(".json")),
            "report": {
                "jerk_profiles": {"generator": np.full((32, 57), level).tolist()},
                "per_pair_diagnostics": {"generator": [{"foot_skate_m_per_s": skate}] * 32},
            },
            "semantic": {"early": {"generator": {"score": np.full(32, level).tolist()}}},
        }
    comparisons = probe.history_contrasts({"mamba": {"0": variants}})["mamba"]["0"]
    assert comparisons["8"]["semantic"]["early"]["score"]["last_minus_first"] == -0.5
    assert comparisons["8"]["physical"]["peak_jerk"]["difference"] == -0.5
    assert comparisons["8"]["physical"]["auj"]["difference"] == 28.5
    assert comparisons["8"]["contact_skate"]["last_minus_first"] == pytest.approx(0.1)
    assert comparisons["8"]["jointly_contacting_pairs"] == 32
    assert comparisons["0"]["contact_skate"] is None
    assert comparisons["0"]["jointly_contacting_pairs"] == 0


@pytest.mark.parametrize(
    "offset,keep,scenario",
    [
        (64, 8, "A_to_B"),
        (0, 16, "A_to_A"),
        (0, 16, "B_to_A"),
    ],
)
def test_history_case_converts_in_memory_tuple_tokens_for_guo(
    monkeypatch,
    tmp_path,
    offset,
    keep,
    scenario,
):
    source = {
        "config": "config",
        "checkpoint": "generator",
        "tokenizer_checkpoint": "tokenizer",
        "backbone": "mamba",
    }
    report = {
        "pairs": [{"left": {"tokens": ("walk/VERB",)}, "right": {"tokens": ("run/VERB",)}}] * 32
    }

    def inputs(tokens):
        assert isinstance(tokens, list)
        return tokens

    resources = SimpleNamespace(text_embedder=None, device="cpu", build_guo_text_inputs=inputs)
    monkeypatch.setattr(probe, "transition_screen", lambda *args: report)
    monkeypatch.setattr(probe, "compute_guo_text_embeddings", lambda *args: np.zeros((32, 1)))
    monkeypatch.setattr(
        probe, "score_switch_archive", lambda *args: ({}, np.zeros((3, 2, 32, 1)), {})
    )
    monkeypatch.setattr(probe, "atomic_numpy_save", lambda *args: None)
    case = probe.history_case(
        source,
        offset,
        keep,
        tmp_path / "case.json",
        None,
        None,
        SimpleNamespace(resources=resources),
        [],
        scenario=scenario,
    )
    assert case["report"] is report


def test_switch_archive_windows_are_exact_and_full_action_reproduces_r6(tmp_path):
    source = tmp_path / "mamba.json"
    folder = source.with_suffix("")
    folder.mkdir()
    features = np.zeros((32, 384, 263), dtype=np.float32)
    features[:, :, 0] = np.arange(384)
    for method in ("cut", "generator"):
        np.save(folder / f"{method}_features.npy", features)
    calls = []
    new = np.arange(32)[:, None] * 10.0 + 2.0
    text = np.stack((new - 2, new))

    def embed(clips):
        calls.append((len(clips[0]), clips[0][0, 0], clips[0][-1, 0]))
        return new + 0.25

    previous = {"per_pair_hits": [1] * 32, "text_distance": 0.25}
    report = {"backbone": "mamba", "retrieval": {"2": {"cut": previous, "generator": previous}}}
    measured, embeddings, hashes = probe.score_switch_archive(
        source, report, SimpleNamespace(compute_motion_embeddings=embed), text
    )
    assert calls == [(192, 192, 383)] * 2 + [(96, 192, 287)] * 2 + [(96, 288, 383)] * 2
    assert embeddings.shape == (3, 2, 32, 1)
    assert measured["late"]["generator"]["new_r_top1"].mean() == 1
    assert hashes["cut"] == hashes["generator"]
    previous["text_distance"] = 99
    with pytest.raises(RuntimeError, match="not reproduced"):
        probe.score_switch_archive(
            source, report, SimpleNamespace(compute_motion_embeddings=embed), text
        )


@pytest.mark.parametrize("invalid", ["short", "integer", "nonfinite"])
def test_switch_archive_rejects_invalid_raw_features(tmp_path, invalid):
    source = tmp_path / "mamba.json"
    source.with_suffix("").mkdir()
    frames = 383 if invalid == "short" else 384
    dtype = np.int16 if invalid == "integer" else np.float32
    features = np.zeros((32, frames, 263), dtype=dtype)
    if invalid == "nonfinite":
        features[0, 0, 0] = np.nan
    np.save(source.with_suffix("") / "cut_features.npy", features)
    with pytest.raises(ValueError, match="finite raw features"):
        probe.score_switch_archive(source, {}, None, np.zeros((2, 32, 1)))


def switch_reports(tmp_path):
    checkpoint = tmp_path / "weights.pt"
    checkpoint.write_bytes(b"frozen")
    common = {
        "pairs": [{}] * 32,
        "config": str(checkpoint),
        "tokenizer_checkpoint": str(checkpoint),
        "checkpoint": str(checkpoint),
        "config_sha256": probe.sha256_file(checkpoint),
        "tokenizer_sha256": probe.sha256_file(checkpoint),
        "checkpoint_sha256": probe.sha256_file(checkpoint),
        "split": "val",
        "seed": 2026,
        "precision": "tf32",
        "sampling": {},
        "fps": 20,
        "action_frames": 192,
        "batch_size": 2,
        "anchor_every": 8,
        "anchor_context": 16,
    }
    paths = []
    for backbone in ("mamba", "transformer"):
        path = tmp_path / f"{backbone}.json"
        path.write_text(json.dumps({**common, "backbone": backbone}), encoding="utf-8")
        paths.append(path)
    return paths, checkpoint


def test_switch_sources_validate_hashes_and_twin_metadata(tmp_path):
    paths, checkpoint = switch_reports(tmp_path)
    assert len(probe.switch_sources(paths)) == 2
    report = json.loads(paths[1].read_text())
    report["seed"] = 123
    paths[1].write_text(json.dumps(report))
    with pytest.raises(ValueError, match="seed"):
        probe.switch_sources(paths)
    report["seed"] = 2026
    paths[1].write_text(json.dumps(report))
    checkpoint.write_bytes(b"changed")
    with pytest.raises(ValueError, match="provenance"):
        probe.switch_sources(paths)


def test_switch_mode_requires_archived_reports_without_loading_a_generator(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["probe", "--caption_switch"])
    with pytest.raises(SystemExit) as caught:
        probe.main()
    assert caught.value.code == 2


def test_bridges_preserve_length_and_all_frames_outside_the_declared_interval():
    cut = np.zeros((100, 22, 3))
    cut[50:, :, 0] = 2.0
    linear = probe.bridge(cut, 50, 20, "linear")
    smooth = probe.bridge(cut, 50, 20, "minimum_jerk")
    for result in (linear, smooth):
        assert result.shape == cut.shape
        np.testing.assert_array_equal(result[:40], cut[:40])
        np.testing.assert_array_equal(result[60:], cut[60:])
        assert np.all(np.diff(result[39:61, 0, 0]) >= 0)
    assert smooth[40, 0, 0] < linear[40, 0, 0]
    assert smooth[59, 0, 0] > linear[59, 0, 0]
    assert probe.jerk_profile(smooth[20:80]).max() < probe.jerk_profile(cut[20:80]).max()


def test_auj_penalizes_the_wrong_jerk_level_even_when_peak_jerk_is_zero():
    reference = np.full((2, 2, 5), 2.0)
    profiles = {"cut": np.ones((2, 5)), "minimum_jerk": np.zeros((2, 5))}
    summary, difference = probe.transition_scores(profiles, reference)
    assert summary["cut"]["peak_jerk"]["value"] == 1
    assert summary["cut"]["auj"]["value"] == 5
    assert summary["minimum_jerk"]["auj"]["value"] == 10
    assert difference["minimum_jerk"]["auj"] == {
        "difference": 5.0,
        "ci95_low": 5.0,
        "ci95_high": 5.0,
    }


def test_peak_is_taken_after_averaging_profiles_not_before():
    profiles = {"cut": np.array([[4.0, 0.0], [0.0, 4.0]])}
    summary, _ = probe.transition_scores(profiles, np.zeros((2, 2, 2)))
    assert summary["cut"]["peak_jerk"]["value"] == 2.0
    assert summary["cut"]["auj"]["value"] == 4.0


def test_transition_scores_reject_an_empty_reference_profile():
    with pytest.raises(ValueError, match="reference jerk"):
        probe.transition_scores({"cut": np.zeros((2, 0))}, np.zeros((2, 2, 0)))


def test_transition_window_must_leave_reference_frames_outside_each_action():
    args = Namespace(prompts=32, segment_tokens=8, boundary_frames=64)
    with pytest.raises(ValueError, match="reference frames"):
        probe.transition_screen(
            args, None, SimpleNamespace(downsample=4), SimpleNamespace(resources=None), []
        )


def test_transition_generation_pairs_sampling_draws_and_reuses_the_left_tokens(tmp_path):
    calls = []

    def stream(text, steps, **kwargs):
        calls.append(kwargs)
        anchors = kwargs["anchors"]
        for index in range(steps):
            condition = anchors.text_at(index) if anchors.text_at else text
            yield condition.long().expand(-1, 2)

    model = SimpleNamespace(cfg=SimpleNamespace(num_codebooks=2), stream_token_indices=stream)
    network = SimpleNamespace(
        decode=lambda indices: (
            indices[:, :, :1].repeat_interleave(4, dim=1).expand(-1, -1, 263).float()
        )
    )
    pipeline = SimpleNamespace(
        token_generator=model,
        tokenizer=SimpleNamespace(network=network),
        downsample=4,
        text_encoder=lambda captions: torch.tensor(
            [[1.0 if caption.startswith("left") else 2.0] for caption in captions]
        ),
    )
    resources = SimpleNamespace(device="cpu", denormalize=lambda clip: clip)
    pairs = [
        (
            probe.ProbePrompt(str(i), str(i), f"left {i}", ()),
            probe.ProbePrompt(str(i + 2), str(i + 2), f"right {i}", ()),
        )
        for i in range(2)
    ]
    features, elapsed = probe.transition_generate(
        pipeline,
        resources,
        pairs,
        48,
        2,
        evidence_path=tmp_path / "original",
    )
    assert elapsed >= 0 and len(calls) == 2
    torch.testing.assert_close(calls[0]["uniforms"][:, 48:], calls[1]["uniforms"])
    assert calls[0]["sampling"].cfg_scale == 6 and calls[1]["anchors"].context == 16
    assert all(clip.shape == (384, 263) for clips in features.values() for clip in clips)
    np.testing.assert_array_equal(features["generator"][0], features["cut"][0])
    probe.transition_generate(
        pipeline,
        resources,
        [(left, left) for left, _ in pairs],
        48,
        2,
        switch_context=0,
        sampling_pairs=pairs,
        evidence_path=tmp_path / "same",
    )
    torch.testing.assert_close(calls[0]["uniforms"], calls[2]["uniforms"])
    np.testing.assert_array_equal(
        np.load(tmp_path / "original" / "uniforms.npy"), np.load(tmp_path / "same" / "uniforms.npy")
    )
    assert np.load(tmp_path / "same" / "indices.npy").shape == (2, 96, 2)
    assert np.load(tmp_path / "same" / "right_indices.npy").shape == (2, 48, 2)
    assert calls[2]["anchors"].context_at(48) == 0
    with pytest.raises(ValueError, match="sampling identities"):
        probe.transition_generate(pipeline, resources, pairs, 48, 2, sampling_pairs=[])


def test_transition_sampling_rejects_test_split_and_unaligned_caption_switches():
    args = Namespace(
        split=Split.TEST,
        family_index=Path("index.csv"),
        save_motion=True,
        segments=2,
        segment_tokens=48,
        blend_frames=20,
        boundary_frames=60,
        policies=None,
    )
    with pytest.raises(ValueError, match="split val"):
        probe.validate_transition_args(args)
    args.split = Split.VALIDATION
    args.segment_tokens = 49
    with pytest.raises(ValueError, match="multiple of eight"):
        probe.validate_transition_args(args)


def test_transition_smoothing_can_shrink_bones_and_the_diagnostic_detects_it():
    pose = np.zeros((22, 3))
    for parent, child in kinematic_bones():
        pose[child] = pose[parent] + RAW_OFFSETS[child] * 0.1
    cut = np.repeat(pose[None], 100, axis=0)
    cut[50:, :, [0, 2]] *= -1
    smooth = probe.bridge(cut, 50, 20, "minimum_jerk")
    deviation = probe.transition_statistics(smooth, 50)["bone_relative_deviation"]
    assert deviation is not None and deviation > 0.01


def test_position_budget():
    full = MemoryPolicy("full_state", 0, 0)
    reset = MemoryPolicy("reset", 8, 16)
    assert policy_fits(full, None, 1, 294)
    assert not policy_fits(full, 96, 1, 294)
    assert policy_fits(full, 96, 1, 95)
    assert policy_fits(reset, 96, 1, 294)
    assert not policy_fits(reset, 24, 1, 294)


def test_paired_interval():
    values = np.array([[1.0, 3.0, 2.0], [2.0, 4.0, 3.0]])
    assert paired_change(values) == {
        "last_minus_first": 1.0,
        "ci95_low": 1.0,
        "ci95_high": 1.0,
    }
    assert paired_change(values) == paired_change(values)


@pytest.mark.parametrize("values", [np.ones(3), np.ones((1, 3)), np.full((2, 3), np.nan)])
def test_invalid_pairs(values):
    with pytest.raises(ValueError):
        paired_change(values)


def test_validation_selection(tmp_path: Path):
    for i in range(5):
        (tmp_path / f"{i}.txt").write_text(f"caption {i}#word/NOUN#0#0", encoding="utf-8")
    splits = []
    resources = SimpleNamespace(
        annotation_dir=tmp_path,
        clip_ids=lambda split: splits.append(split) or [str(i) for i in range(5)],
    )
    first = probe.test_prompts(resources, 3, Split.VALIDATION)
    second = probe.test_prompts(resources, 3, Split.VALIDATION)
    assert first == second
    assert splits == [Split.VALIDATION, Split.VALIDATION]
    assert len(first) == 3
    assert all(tokens == ["word/NOUN"] for _, tokens in first)


def test_missing_caption(tmp_path: Path):
    resources = SimpleNamespace(annotation_dir=tmp_path, clip_ids=lambda split: ["missing"])
    with pytest.raises(FileNotFoundError):
        probe.test_prompts(resources, 1, Split.VALIDATION)


def test_source_family_cohorts(tmp_path: Path):
    index = tmp_path / "index.csv"
    index.write_text(
        "source_path,new_name\n" + "".join(f"recording_{i // 2}.npy,{i}.npy\n" for i in range(128)),
        encoding="utf-8",
    )
    for i in range(128):
        (tmp_path / f"{i}.txt").write_text(f"caption {i}#word/NOUN#0#0", encoding="utf-8")
    resources = SimpleNamespace(
        annotation_dir=tmp_path,
        clip_ids=lambda split: [str(i) for i in range(128)] if split == Split.VALIDATION else [],
    )
    first = probe.family_prompts(resources, 32, Split.VALIDATION, index, 0)
    second = probe.family_prompts(resources, 32, Split.VALIDATION, index, 32)
    assert len({item.source_family for item in first}) == 32
    assert not {item.source_family for item in first} & {item.source_family for item in second}
    assert first == probe.family_prompts(resources, 32, Split.VALIDATION, index, 0)
    assert all(item.tokens == ("word/NOUN",) for item in first)


def test_missing_family_mapping_fails(tmp_path: Path):
    index = tmp_path / "index.csv"
    index.write_text("source_path,new_name\n", encoding="utf-8")
    resources = SimpleNamespace(annotation_dir=tmp_path, clip_ids=lambda split: ["missing"])
    with pytest.raises(KeyError):
        probe.family_prompts(resources, 32, Split.VALIDATION, index, 0)


def test_family_selection_excludes_training_recordings(tmp_path: Path):
    index = tmp_path / "index.csv"
    index.write_text(
        "source_path,new_name\nrecording_0.npy,train.npy\n"
        + "".join(f"recording_{i}.npy,{i}.npy\n" for i in range(40)),
        encoding="utf-8",
    )
    for i in range(40):
        (tmp_path / f"{i}.txt").write_text(f"caption {i}#word/NOUN#0#0", encoding="utf-8")
    resources = SimpleNamespace(
        annotation_dir=tmp_path,
        clip_ids=lambda split: ["train"] if split == Split.TRAIN else [str(i) for i in range(40)],
    )
    samples = probe.family_prompts(resources, 32, Split.VALIDATION, index, 0)
    assert all(sample.source_family != "recording_0.npy" for sample in samples)


def test_comparison_evidence(tmp_path: Path):
    paths = [tmp_path / "mamba.json", tmp_path / "transformer.json"]
    for backbone, path in zip(("mamba", "transformer"), paths, strict=True):
        report = {
            "backbone": backbone,
            "captions": [f"caption {i}" for i in range(32)],
            "prompts": 32,
            "split": "val",
            "seed": 2026,
            "sampling": {"cfg_scale": 6.0},
            "config": "config",
            "tokenizer_checkpoint": "tokenizer",
            "precision": "tf32",
            "batch_size": 4,
            "segment_tokens": 49,
            "policies": {},
            "per_prompt": {},
        }
        policies = (
            ("full_state", "reset_8_keep_16", "reset_32_keep_16")
            if backbone == "mamba"
            else ("reset_8_keep_16", "reset_32_keep_16")
        )
        for policy in policies:
            late = 2.0 if policy == "full_state" else 1.5 if backbone == "mamba" else 1.2
            report["policies"][policy] = [
                {"segment": 1, "seconds_from": 0.0, "text_distance": 1.0, "r_top1": 1.0},
                {"segment": 2, "seconds_from": 9.8, "text_distance": late, "r_top1": 1.0},
            ]
            report["per_prompt"][policy] = {
                "text_distance": [[1.0] * 32, [late] * 32],
                "r_top1": [[1.0] * 32] * 2,
            }
        path.write_text(json.dumps(report), encoding="utf-8")
    out = tmp_path / "comparison.json"
    csv = tmp_path / "summary.csv"
    probe.compare(paths, out, csv)
    comparison = json.loads(out.read_text(encoding="utf-8"))["comparisons"]["text_distance"]
    assert comparison["mamba_restart_minus_full_late"]["last_minus_first"] == -0.5
    assert np.isclose(comparison["mamba_minus_transformer_restart_late"]["last_minus_first"], 0.3)
    assert len(csv.read_text(encoding="utf-8").splitlines()) == 11
    assert comparison["mamba_32_minus_8_late"]["last_minus_first"] == 0.0
    with pytest.raises(FileExistsError):
        probe.compare(paths, out, csv)
