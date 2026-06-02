# Module Development Research
*Context: M1 (flan-T5-small) is training -- eval_loss 0.0719 at epoch 4 -> targeting < 0.05.*
*This document covers library options and upgrade paths for M2, M4, M5, M6, M8 once M1 is stable.*

---

## M2 -- Scene Planner

### Current State
`src/modules/planner/planner.py` is a **pure rules-based** spatial layout engine.
It hand-codes actor/object placement using fixed constants (`_ACTOR_DIST = 1.5`, `_OBJ_SPACE = 0.4`,
`_FALL_H = 1.5`). Logic handles ~3 action types (`fall`, `kick`, `pick_up`). Cannot generalize
to new spatial relations or complex multi-agent scenes.

### Problem
Given M1 output like `{"entities": [...], "actions": [...], "relations": [...]}`, we need to:
1. Convert entity names + spatial relations into valid 3-D coordinates for PyBullet
2. Avoid object overlap / interpenetration
3. Place things semantically: "a ball on the table" != "a ball next to the table"

### Library Options

| Library / Approach | Notes |
|--------------------|-------|
| **Constraint-satisfaction (z3 / scipy.optimize)** | Express "on top of", "left of", etc. as inequality constraints, solve numerically. No ML needed. **Best short-term option.** |
| **ATISS** (Autoregressive Transformers for Indoor Scene Synthesis) | Transformer trained on ScanNet/3D-FRONT to generate furniture layout. Overkill for our use case (interior design focused). |
| **LayoutGPT** (Cheng et al., 2023 -- `arxiv 2305.15393`) | LLM prompted to output JSON spatial layouts. Requires GPT-4/Llama-70B. Interesting for future. |
| **SceneScript** (Apple Research, 2024) | Autoregressive structure language for room-scale 3D scenes. Heavy -- not suitable for real-time. |
| **LLM spatial prompting** (local Llama 3 8B) | Add a `spatial_llm` option to ScenePlanner: send entities + relations as JSON, receive positions back. Fast with `llama.cpp` / `ollama`. **Best medium-term option.** |

### Recommended Upgrade Path

**Phase 1 (now):** Replace hard-coded constants with a `ConstraintLayout` class using
`scipy.optimize.minimize` to satisfy:
- `on(A, B)` -> `z_A ~= z_B + h_B / 2 + h_A / 2`
- `left_of(A, B)` -> `x_A < x_B  gap`
- `near(A, B)` -> `|pos_A  pos_B| < threshold`

```python
# Sketch
from scipy.optimize import minimize
import numpy as np

def layout_loss(positions_flat, constraints):
    positions = positions_flat.reshape(-1, 3)
    loss = 0.0
    for c in constraints:
        if c.type == "on":
            dz = positions[c.a][2] - (positions[c.b][2] + c.b_half_h + c.a_half_h)
            loss += dz ** 2
        elif c.type == "left_of":
            loss += max(0, positions[c.a][0] - positions[c.b][0] + 0.3) ** 2
    return loss
```

**Phase 2 (after M1 stable):** Integrate `ollama` with a local Llama 3 8B to produce
spatial JSON from M1 relation text. One call, sub-second on CPU.

```
pip install ollama
```

---

## M4 -- Motion Generator

### Current State
Three-backend system (see `generator.py`):
1. **Retrieval** -- random-choice from KIT-ML index (good quality, limited variety)
2. **SSM model** -- trained `checkpoints/motion_ssm/best_model.pt` (medium quality)
3. **Placeholder** -- procedural fallback (poor quality)

KIT-ML motions are 251-dim (joints x 3 + root velocity). The SSM model generates
frames autoregressively. The retrieval backend only does keyword matching
(`walk`, `run`, `jump`, ...) with no semantic understanding.

### Problem
- Retrieval returns random clips -- no text-motion alignment
- SSM model quality is adequate for demos but not evaluation-grade
- No SMPL / HumanML3D standard output (makes comparison with SOTA hard)

### Library Options

| Library | Quality | VRAM | Notes |
|---------|---------|------|-------|
| **MDM** (Motion Diffusion Model, Tevet et al. 2022) |  | 8 GB | Pretrained on HumanML3D. Outputs SMPL joints. `pip install` available. |
| **MLD** (Motion Latent Diffusion, Chen et al. 2023) |  | 6 GB | Faster than MDM, same quality. VAE latent space. |
| **MotionDiffuse** (Zhang et al. 2022) |  | 6 GB | Part-based diffusion, good fine-graining. |
| **T2M-GPT** (Zhang et al. 2023) |  | 4 GB | VQ-VAE + GPT. Fastest inference of diffusion alternatives. |
| **FLAME / SMPL-X** | -- | CPU | Body model needed downstream for rendering |
| **human_body_prior** | -- | CPU | VPoser PCA prior -- useful for motion regularization |
| **BVH / KIT-ML adapter** (current) |  | 0 | Already integrated. Keep as fallback. |

### Recommended Upgrade Path

**RTX 3050 (4 GB VRAM) constraint** -> MDM full model is borderline (8 GB).
Use **T2M-GPT** as the primary upgrade target -- runs in ~3 GB.

```bash
# T2M-GPT (HumanML3D 263-dim output, easy SMPL conversion)
git clone https://github.com/Mael-zys/T2M-GPT
pip install -r T2M-GPT/requirements.txt

# Or: use the HuggingFace MDM port (lighter)
pip install git+https://github.com/GuyTevet/motion-diffusion-model
```

**Integration sketch** -- add a `DiffusionMotionBackend` to `generator.py`:

```python
class DiffusionMotionBackend:
    """T2M-GPT or MDM backend for high-quality text-to-motion."""

    def __init__(self, model_path: str = "checkpoints/t2m_gpt"):
        self._model = None
        self._load(model_path)

    def generate(self, text: str, max_frames: int = 200) -> Optional[MotionClip]:
        if self._model is None:
            return None
        with torch.no_grad():
            motion = self._model.generate(text, num_frames=max_frames)   # (T, 263)
        frames = self._huml3d_to_kit(motion)   # convert 263->251 dims
        return MotionClip(action=text, frames=frames, source="diffusion")

    @staticmethod
    def _huml3d_to_kit(motion: np.ndarray) -> np.ndarray:
        # HumanML3D is 263-dim, KIT-ML is 251-dim
        # Both represent root + joint velocities -- mapping documented in
        # HumanML3D repo `data/dataset.py`
        ...
```

**Semantic retrieval improvement** (low-hanging fruit, no extra model needed):
Replace keyword matching with SBERT cosine similarity:

```python
from sentence_transformers import SentenceTransformer

class SemanticRetriever(MotionRetriever):
    def __init__(self, data_dir="data/KIT-ML"):
        super().__init__(data_dir)
        self._sbert = SentenceTransformer("all-MiniLM-L6-v2")  # 80 MB
        self._embeddings = self._sbert.encode(
            [s.text for s in self._samples], convert_to_tensor=True
        )

    def retrieve(self, text: str, max_frames: int = 200) -> Optional[MotionClip]:
        q = self._sbert.encode(text, convert_to_tensor=True)
        scores = util.cos_sim(q, self._embeddings)[0]
        idx = int(scores.argmax())
        s = self._samples[idx]
        return MotionClip(action=text, frames=s.motion[:max_frames], source="semantic_retrieved")
```

```bash
pip install sentence-transformers
```

---

## M5 -- Physics Engine

### Current State
Uses **PyBullet** for rigid-body simulation. The `CinematicCamera` and video export already
work. Physics accuracy is good for stiff contacts but PyBullet has known issues:
- Unstable soft-body / cloth simulation
- No GPU acceleration
- Contact solver can tunnel fast-moving objects

### Library Options

| Library | Pros | Cons |
|---------|------|------|
| **PyBullet** (current) | Stable, easy API, pip install | Single-core CPU, legacy |
| **MuJoCo 3** (`mujoco` 3.x) | Best contact accuracy, RL-standard, Apache 2.0 since 2022 | More verbose XML setup |
| **Brax** (Google, JAX-based) | Fully differentiable, GPU-parallel | Requires JAX, less mature |
| **NVIDIA Isaac Gym / Isaac Lab** | GPU physics + RL env, 1000x speedup | Requires NVIDIA Isaac SDK install |
| **Genesis** (2024, MIT) | GPU-accelerated, Python-native, faster than Isaac | Very new, API unstable |

### Recommendation
For the dissertation scope: **keep PyBullet for M5**, but add an optional **MuJoCo adapter**.

```bash
pip install mujoco   # >= 3.0, Apache 2.0, < 5 MB
```

MuJoCo 3 can load URDF files (same as PyBullet) and provides `mjData` physics state
that's needed for RL controller training (M6).

---

## M6 -- RL Controller

### Current State
`controller.py` is a **pure stub**:
```python
class RLController:
    def setup(self): raise NotImplementedError
    def act(self, observation): raise NotImplementedError
```
The docstring says: "wrap M5 Scene as `gymnasium.Env`, train with stable-baselines3 PPO."
This is exactly the right plan.

### Library Options

| Library | Role | Notes |
|---------|------|-------|
| **Stable-Baselines3 (SB3)** | PPO / SAC / TD3 algorithms | Best-documented RL library for PyTorch. Active. |
| **Gymnasium** (successor to OpenAI Gym) | Env interface standard | All SB3 algos expect `gymnasium.Env` |
| **AMP** (Adversarial Motion Priors, Peng et al. 2021) | Learn physically-plausible locomotion from motion capture | Very high quality -- but complex to train |
| **DeepMimic** (Peng et al. 2018) | Motion imitation via RL | Classic approach; SB3 compatible |
| **MotionVAE** | Low-dim latent action space | Reduces PPO action dim from 63 -> ~8 |
| **MuJoCo Humanoid-v4** | Pre-built humanoid env | Good reference baseline |

### Implementation Plan

**Step 1 -- Gymnasium wrapper around M5:**

```python
import gymnasium as gym
import numpy as np
from src.modules.physics.engine import PhysicsScene

class PhysicsSceneEnv(gym.Env):
    """Wraps M5 PhysicsScene as a standard gymnasium environment."""

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, scene_config, target_motion: np.ndarray):
        super().__init__()
        self._config = scene_config
        self._target = target_motion          # from M4
        self._scene: PhysicsScene | None = None

        n_joints = 21  # KIT-ML humanoid DOFs
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(n_joints * 6,), dtype=np.float32
        )  # pos + vel for each joint
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(n_joints,), dtype=np.float32
        )  # joint torques (normalized)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._scene = PhysicsScene(self._config)
        self._frame = 0
        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        self._scene.apply_torques(action)
        self._scene.step()
        self._frame += 1
        obs = self._get_obs()
        reward = self._compute_reward()
        terminated = self._frame >= len(self._target)
        return obs, reward, terminated, False, {}

    def _compute_reward(self) -> float:
        """AMP-style pose matching reward: -||q_sim - q_ref||^2"""
        sim_pose = self._get_joint_positions()
        ref_pose = self._target[min(self._frame, len(self._target) - 1)]
        return float(-np.mean((sim_pose - ref_pose[:len(sim_pose)]) ** 2))

    def _get_obs(self) -> np.ndarray:
        return np.zeros(self.observation_space.shape, dtype=np.float32)  # TODO: real obs

    def _get_joint_positions(self) -> np.ndarray:
        return np.zeros(21, dtype=np.float32)  # TODO: extract from PyBullet body
```

**Step 2 -- Train with SB3 PPO:**

```python
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

env = make_vec_env(lambda: PhysicsSceneEnv(scene_config, target_motion), n_envs=4)
model = PPO(
    "MlpPolicy", env,
    n_steps=2048, batch_size=64, n_epochs=10,
    gamma=0.99, gae_lambda=0.95,
    ent_coef=0.01, vf_coef=0.5,
    verbose=1,
    device="cuda",
)
model.learn(total_timesteps=1_000_000)
model.save("checkpoints/rl_ppo/humanoid_walk")
```

```bash
pip install stable-baselines3 gymnasium
```

**Expected training time on RTX 3050:** ~2-4 hours for basic locomotion.

### Reward Shaping (AMP-lite)
Instead of just pose matching, combine:

```
R_total = w1 * R_pose      # match M4 reference motion joints
        + w2 * R_energy     # penalize large torques  -> smoother motion
        + w3 * R_alive      # reward for not falling
        + w4 * R_velocity   # match root velocity from M4
```

---

## M8 -- AI Enhancer / Video Stylizer

### Current State
Uses ControlNet (skeleton -> image). Status: optional, stub-level integration.

### Library Options

| Library | Notes |
|---------|-------|
| **AnimateDiff** (Guo et al. 2023) | Adds temporal consistency to SD 1.5. Good for short clips. `pip install animatediff-cli` |
| **Stable Video Diffusion (SVD)** | Stabilityai, image->video. 4 GB VRAM with attention slicing. |
| **CogVideoX** (ZhipuAI, 2024) | State-of-the-art text+image -> video. 5B params, needs 16 GB VRAM ideally -- too heavy. |
| **IP-Adapter** | Style/image conditioning for SD. Lightweight. |
| **ControlNet** (current) | Skeleton/depth -> image. Per-frame only, no temporal. |
| **diffusers** library | Unified HuggingFace API for all of the above. |

### Recommendation
Use `diffusers` as the unified backend -- it supports AnimateDiff, SVD, and ControlNet
with the same API surface:

```bash
pip install diffusers accelerate transformers
```

```python
# AnimateDiff example (consistent video from frames)
from diffusers import AnimateDiffPipeline, MotionAdapter, EulerDiscreteScheduler
from diffusers.utils import export_to_video

adapter = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-v1-5-2")
pipe = AnimateDiffPipeline.from_pretrained(
    "emilianJR/epiCRealism",
    motion_adapter=adapter,
    torch_dtype=torch.float16,
).to("cuda")

frames = pipe(
    prompt="a person walking in a park, cinematic, 4k",
    negative_prompt="blurry, distorted",
    num_frames=16,
    guidance_scale=7.5,
).frames[0]

export_to_video(frames, "outputs/videos/enhanced.mp4", fps=8)
```

**Memory note:** AnimateDiff with `enable_attention_slicing()` runs in ~4 GB VRAM.

---

## Suggested Development Order (after M1 stable)

```
M1 stable (eval_loss < 0.05)
        |
        + M2: Add ConstraintLayout (scipy)         <- 1-2 days
        |         then LLM spatial prompting
        |
        + M4: Add SemanticRetriever (SBERT)         <- 1 day
        |         then T2M-GPT backend
        |
        + M5: Add optional MuJoCo adapter          <- 1 day
        |         (needed for M6 RL training)
        |
        + M6: Implement PhysicsSceneEnv + PPO       <- 3-5 days
        |         train on RTX 3050 (~3h)
        |
        + M8: Wire AnimateDiff via diffusers         <- 1 day
```

---

## Quick Install Summary

```bash
# M2 improvements
pip install scipy ollama

# M4 improvements
pip install sentence-transformers

# M5 improvements
pip install mujoco

# M6 (RL)
pip install stable-baselines3 gymnasium

# M8 (video AI)
pip install diffusers accelerate

# Shared
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

---

*Last updated: during M1 v5 training run -- step ~10k/37.5k, ~28%, ETA ~6h.*
