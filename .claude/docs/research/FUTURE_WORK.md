# Future Implementations Research

Research summary for remaining dissertation pipeline features.

---

## 1. Motion Generation Alternatives

### Current: MDM (Motion Diffusion Model)

- Paper: <https://arxiv.org/abs/2209.14916>
- Quality: State-of-the-art
- Speed: ~8s per motion
- Checkpoint: ~1GB

### Faster Alternatives

| Model | Speed | Quality | Link |
|-------|-------|---------|------|
| **MLD** | 2 orders faster | Same quality | [GitHub](https://github.com/ChenFengYe/motion-latent-diffusion) |
| **T2M-GPT** | Very fast | Competitive | [GitHub](https://mael-zys.github.io/T2M-GPT/) |
| **SATO** (2024) | Fast | More stable | <https://arxiv.org/abs/2405.01461> |

### Recommendation

Start with **MLD** - same quality as MDM but 100x faster. Best for iterative development.

---

## 2. Physics-Based Character Control

### PULSE (ICLR 2024 Spotlight)

**Universal Humanoid Motion Representations for Physics-Based Control**

- Paper: <https://arxiv.org/abs/2310.04582>
- Code: <https://github.com/ZhengyiLuo/PULSE>
- What it does: Distills motion skills into latent space for physics control
- Use case: Makes humanoid robust to perturbations

### Key Features

- Train once -> use for any motion
- Works with PyBullet/Isaac Gym
- Handles falls and recovery

### When to Use

If you need humanoid to **interact with environment** (push objects, get pushed), not just playback motion.

---

## 3. RL Controller (Module 6)

### Setup with Stable-Baselines3

```python
from stable_baselines3 import PPO
import gymnasium as gym

# Create environment
env = gym.make("Humanoid-v4")

# Train with PPO
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=1_000_000)
```

### When You Need RL

- Walking on uneven terrain
- Recovering from pushes
- Object manipulation (grasping)

### Training Time

| Task | Timesteps | Time (RTX 3080) |
|------|-----------|-----------------|
| Basic walk | 1M | ~2 hours |
| Robust walk | 10M | ~20 hours |
| Complex tasks | 50M+ | 1-3 days |

### Alternative: Use Existing Policies

- [Humanoid-Gym](https://github.com/roboterax/humanoid-gym) - Pre-trained walking
- [SB3 RL Zoo](https://github.com/DLR-RM/rl-baselines3-zoo) - Pre-trained models

---

## 4. Physical Diffusion Policy (PDP)

Latest approach merging diffusion + physics (Dec 2024):

- Paper: <https://arxiv.org/abs/2412.17238>
- What: Use diffusion for motion AND physics control
- Benefit: Handles unstable bipedal walking better

### Trade-off

More complex to implement, but more robust than kinematic playback.

---

## 5. Recommended Implementation Order

| Priority | Feature | Effort | Impact |
|----------|---------|--------|--------|
| 1 | Connect motion to humanoid | Low | High |
| 2 | Replace MDM with MLD | Medium | High (faster) |
| 3 | Add simple RL for recovery | Medium | Medium |
| 4 | Integrate PULSE | High | Very High |

---

## 6. Quick Wins

### A. Test humanoid body loading

```bash
py -c "from src.modules.physics.humanoid import HumanoidBody; print('OK')"
```

### B. Apply motion to humanoid (next step)

Modify pipeline to:

1. Load humanoid in physics scene
2. Apply motion frames each timestep
3. Render result

### C. Enable ControlNet for final demo

Already built in Module 8, just enable flag.

---

## References

1. **MDM**: <https://arxiv.org/abs/2209.14916> (ICLR 2023)
2. **MLD**: <https://arxiv.org/abs/2212.04048> (CVPR 2023)
3. **T2M-GPT**: <https://arxiv.org/abs/2301.06052> (CVPR 2023)
4. **PULSE**: <https://arxiv.org/abs/2310.04582> (ICLR 2024)
5. **PDP**: <https://arxiv.org/abs/2412.17238> (Dec 2024)
6. **Stable-Baselines3**: <https://stable-baselines3.readthedocs.io/>
7. **Humanoid-Gym**: <https://github.com/roboterax/humanoid-gym>
