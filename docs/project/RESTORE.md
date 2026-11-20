# Restore on the new laptop

1. Extract `thessis\` to `C:\Users\<you>\Documents\thessis`.
2. Extract `.claude\` and `.copilot\` to `C:\Users\<you>\`.
3. If the user name is not `ic.butacu`: edit the hook paths in `.claude\settings.json`, and rename the folder
   `.claude\projects\C--Users-ic-butacu-Documents-thessis` to match the new repo path.
4. Install uv, then run `uv sync` in `thessis`. Install the Mamba kernel from `thessis\dist\mamba_kernel`.
5. Log in again: `claude`, `copilot`, `gh auth login`. Reinstall the Claude plugins from `.claude\plugins\installed_plugins.json`.
6. Download the data again into `thessis\data` (HumanML3D, AMASS, Motion-X, GloVe, official evaluator, SMPL-X, BABEL),
   then rebuild the derived folders (HumanML3D_263, HumanML3D_60fps, AMASS_263) with `scripts\dataset`.

   The backup keeps our own small data files: splits, Mean/Std stats, captions, clean lists in `thessis\data`,
   and the old `D:\Facultate\dissertation` notes and results in `dissertation\`. Put them back over the downloads.

Not in this backup: data, checkpoints, .venv, caches, infra\*.exe, artifacts\outputs\*.npy, Copilot session-state, credentials.
