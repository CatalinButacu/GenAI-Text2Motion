# Restore TODO (new laptop, 2026-11-18)

Status of each item: [ ] open, [~] in progress, [x] done.

## 0. GPU driver (blocks every GPU job)
- [ ] The new laptop has an RTX 4050 Laptop (6 GB, sm_89) with driver 566.07, which supports at most CUDA 12.7.
      The locked torch is `2.12.0+cu132` (CUDA 13.2). Install an NVIDIA driver >= 591.44 (the version in
      pyproject.toml) before `uv sync`, or torch will not see the GPU.
- [ ] The latency/memory benchmarks were measured on the old RTX 3050 4 GB laptop. Do not mix new timings with them.

## 1. Tools (nothing runs without these)
- [ ] Install Git, uv, Python 3.12 (only the Microsoft Store alias for Python exists now), GNU make, xelatex + biber.
- [ ] Move the repo to `C:\Users\ic.butacu\Documents\thessis` (RESTORE.md step 1), or edit the hook paths in
      `.claude\settings.json`. They point there, and that folder does not exist.
- [ ] `uv sync --inexact --extra dev --extra prep`, then `make kernel-restore` (`dist\mamba_kernel` survived).
- [ ] `make check`.

## 2. Data
Public sources. Each one matches the sha256 and the commit in the old HuggingFace download caches.
- [x] HumanML3D 263 features downloaded: `TeoGchx/HumanML3D` @ 9265d4c3 -> `data\hml3d_official\data\*.parquet`
      (10 files, 4.06 GB, sha256 OK). The rows per split (23384/1460/4384) equal the backed-up split files.
- [ ] Convert the parquet (needs Python + `--extra prep`). The converter OVERWRITES train/val/test.txt and Mean/Std.npy in its output folder,
      so convert into a temporary folder and copy only `new_joint_vecs\`:
      `text2motion prepare --source_release data\hml3d_official --converted_dir data\_hml3d_convert`
      Copy `data\_hml3d_convert\new_joint_vecs` to `data\HumanML3D_official\`, then compare the hashes against
      `data\_restore_backup\SHA256SUMS.txt`.
- [x] Guo evaluator: `camenduru/MoMask` @ 349c4f5f `humanml3d_evaluator.zip` (sha256 OK), extracted to
      `data\official_evaluator\extracted\`. The 14 surviving evaluator files are byte-identical to the new copies,
      and `data\eval_stats\{mean,std}.npy` equal `Comp_v6_KLD005\meta\`. It is the same evaluator as before.
- [ ] Fix `.env.example` / Docker: it uses `extracted\t2m\text_mot_match\...`, but the zip has no `t2m\` level
      (configs/default.yaml is correct).
- [x] Evaluator word vectors: `camenduru/MoMask` `glove.zip` (sha256 OK) -> `data\t2m_glove\glove\our_vab_*`.
- [ ] Gated, needs a login: SMPL-X v1.1 -> `data\smplx_models\smplx\` (see WHAT_GOES_HERE.txt there).
- [ ] Gated, needs a login: AMASS "SMPL-X G" -> `data\amass\<Subset>\` (see WHAT_GOES_HERE.txt there).
      Needed only to rebuild HumanML3D_60fps, AMASS_263 and the pretraining corpus (`data\amass_tokens_fsq8x1024.npz`).
- [ ] Rebuild the derived folders with `scripts\dataset\build_hml3d.py` and `text2motion prepare --corpus`.
- [x] Survived in the backup: HumanML3D texts (29 232), index.csv, splits, Mean/Std, eval_stats, BABEL, extra_motion CSVs.

## 3. Lost: checkpoints (highest risk)
`artifacts\checkpoints\` does not exist. Every tokenizer and generator checkpoint is gone, including the
frozen `fsq_g8_v1024.pt`. A retrained tokenizer invalidates every generator number.
Look for copies before retraining anything:
- [ ] The old laptop or the old `D:\` drive.
- [ ] AWS S3 bucket `thesis-t2m-913402647373` (eu-north-1): `results/`, `bundle.tar` (it holds fsq_g8_v1024.pt).
- [ ] Azure storage container `checkpoints` (infra\azure\main.tf).
- [ ] Kaggle private dataset `catalinbutacu/thesis-code-bundle`.

## 4. Code that points at the old machine
- [x] `scripts\dataset\build_hml3d.py`, `prepare_source.py` and `check_quality.py` now use `REPO / "data" / "amass"`
      instead of `D:\Facultate\dissertation\data\amass`. Run `make check` once Python is installed (not run yet).
- [x] Deleted on 2026-11-18: infra\*\.terraform (942 MB of provider .exe files; `terraform init` restores them),
      src\text2motion.egg-info, .ropeproject, BABEL __MACOSX, imgui.ini, data\glove_clean, data\momask_space,
      ..\dissertation\data\t2m_download.
- [ ] README and `.claude\docs\ARCHITECTURE.md` do not show the `dataset` package. The layer table in
      `tests\test_package_layers.py` is correct.
- [ ] AGENTS.md says "no underscore prefix", but `.claude\hooks\code_rules_check.py` asks for `__name`. Choose one.

## 5. Cloud and secrets
- [ ] `infra\aws\terraform.tfstate` shows `i-042dbd8ab84dfa311` (g5.xlarge, eu-west-1) as **running**. Check the
      AWS console. If it still runs, it costs money: `terraform destroy` or stop it.
- [ ] `infra\aws\dissertation_v2.pem` is a private key in the project folder. It is git-ignored. Keep it out of
      any new backup or zip, or rotate it.
