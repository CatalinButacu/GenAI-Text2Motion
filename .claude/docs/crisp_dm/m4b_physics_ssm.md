# Phase 4b: Physics-Aware SSM Model (CRISP-DM Iteration)

## 1. Business Understanding
**Objective:** Enhance the Text-to-Motion generative model ensuring that avatars interact plausibly with their 3D environment. Pure kinematic models (like the Phase 4a baseline) train effectively but suffer from foot sliding, ground penetration, and physically impossible poses during dynamic actions. 
**Success Criteria:**
1. A measurable reduction in physics-based failure metrics (foot sliding velocity, penetration depth).
2. The model must integrate natively with a differentiable physics engine simulator or evaluate physical losses directly on the SMPL-X sequence without losing text-conditional semantic richness.

## 2. Data Understanding
**Data Format:**
- Ground Truth is purely formulated via AMASS (True 168-dim SMPL-X axis-angle poses at `30.0` fps).
- Text descriptions are mapped from HumanML3D using the `index.csv` explicit frame-segment matcher.
- 263-dim HumanML3D engineered kinematic features are explicitly **disallowed** during training because physics constraints require pure axis-angle rotation tensors (global orientation, body, hands, etc.) to map properly backward through SMPL-X forward kinematics.
- Evaluation against prior literature explicitly uses the separate evaluation subset to match the HumanML3D distribution, while the training corpus is robustly physically grounded.

**Constraints:**
- Physics evaluation requires instantaneous mapping from 168-dim parameters to 3D Cartesian coordinates via `FKExtractor`.
- Motion speed variations are normalized.

## 3. Data Preparation
Our data pre-processing explicitly aligns with SMPL-X requirements:
1. **Dynamic Slicing:** The dataset class now parses the `index.csv` official mappings to slice local `.npz` sequences seamlessly.
2. **Resampling:** The AMASS frames (often at 120 FPS or 60 FPS) are dynamically downsampled to match the standard 30 FPS.
3. **Filtering:** Mirrored poses (M*) are removed from the training projection phase since accurate 3D physics requires complex matrix mirroring of the SMPL-X body skeleton, ensuring no artifacts are injected.

## 4. Modeling
**Architecture:**
- **Text Encoder:** Pre-trained SBERT (`all-MiniLM-L6-v2`, frozen) mappings producing `(B, 384)`.
- **SSM Backbones:** Mamba / Bi-Directional Mamba providing long-context temporal stability without the quadratic cost of Transformers.
- **Physics Layer:** The model integrates an extraction node via `FKExtractor(168-dim)` converting parameters to 3D joints and evaluates specific physical loss terms alongside kinematic distance (MSE, L1, L2).
- **Loss Terms:**
  - `L_mse`: standard rotation parameter difference.
  - `L_foot_slide`: penalty applied when heel/toe velocities exceed thresholds while contacting the ground.
  - `L_penetration`: heavy penalty for joint coordinates `< 0` on the vertical (Y) plane in absolute world space.

## 5. Evaluation
The model is graded continuously against official HumanML3D T2M specifications:
- **FID:** Tested rigorously using the official CVPR 2022 `T2MMotionEncoder` loaded with the `finest.tar` weights.
- **R-Precision @ 1/2/3:** Using 32-sample pool cross-comparison.
- **Physics Robustness:** Comparing before/after metrics on penetration frequency.

## 6. Deployment & Next Steps
- Implement robust parameter sweeping for the Physics loss weighting ($\lambda_{phys}$).
- Produce comparison outputs (side-by-side renders or animations) verifying the physical grounding of the generated motions under dynamic tasks (e.g., walking, jumping, sitting).
