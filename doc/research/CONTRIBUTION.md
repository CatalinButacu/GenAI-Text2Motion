# Your Dissertation Contribution - Research Gap Analysis

> "If I have seen further, it is by standing on the shoulders of giants." - Isaac Newton

## The Giants (Foundation Papers)

| Module | Giant | What They Built | Link |
|--------|-------|-----------------|------|
| Story Agent | Text2Scene, Stanford Spatial | Text -> 2D/3D scene layouts | [CVPR 2019](https://arxiv.org/abs/1809.01110) |
| Model Generator | Shap-E, TripoSR | Text/Image -> 3D mesh | [arXiv 2023](https://arxiv.org/abs/2305.02463) |
| Physics Engine | PyBullet/Bullet | Rigid body dynamics | [pybullet.org](https://pybullet.org) |
| Video Renderer | ControlNet, Stable Diffusion | Depth-conditioned image generation | [ICCV 2023](https://arxiv.org/abs/2302.05543) |

---

## The Gap (What They All Missed)

```
++
|  Text -> Scene Layout -> 3D Meshes -> ???  -> Video        |
|                                           ^                     |
|                              PHYSICS SIMULATION                 |
|                              (Nobody connected these!)          |
++
```

### Current State of Research

| Approach | Physics | Visual Quality | Example |
|----------|---------|----------------|---------|
| Pure AI Video (Sora, etc.) |  Fake | [x] Beautiful | Objects pass through each other |
| Text-to-3D (Shap-E) |  None | [x] Good mesh | Static model, no motion |
| Physics Sim (PyBullet) | [x] Perfect |  Basic | Robot/game graphics |
| ControlNet Video |  None | [x] Realistic | Flickering, no physics |

### YOUR UNIQUE POSITION

| Approach | Physics | Visual Quality | YOUR PROJECT |
|----------|---------|----------------|--------------|
| **Physics-Constrained Video** | [x] Perfect | [x] Enhanced | Physically correct + AI-beautified |

---

## Your Original Contribution

### 1. The Bridge Nobody Built
> **Text -> Physics -> Video**

You are connecting three isolated research areas:
- NLP scene parsing (academic)
- Physics simulation (robotics/games)  
- AI image generation (creative AI)

### 2. Specific Innovations

| Innovation | Why It's Novel | Evidence |
|------------|----------------|----------|
| Action Parsing | "falls" -> gravity force | No paper does verb->physics mapping |
| Physics-to-Depth | True depth from simulation | Others estimate depth from RGB |
| Temporal Grounding | Physics = consistent motion | ControlNet videos flicker without this |

### 3. Your Thesis Statement
> "We present a pipeline that generates physically-accurate video from natural language 
> by combining semantic parsing, physics simulation, and conditional image synthesis, 
> addressing the fundamental limitation of AI video generators: lack of physical realism."

---

## Investigation Roadmap

### Phase 1: Foundation (Current)
- [x] Basic pipeline working
- [ ] Benchmark physics accuracy
- [ ] Document baseline quality

### Phase 2: Your Investigation Areas
1. **Semantic-to-Physics Mapping**
   - Research question: How accurately can we infer physical properties from text?
   - Experiments: Compare inferred mass/friction to ground truth

2. **Temporal Consistency**
   - Research question: Does physics-grounded depth reduce ControlNet flickering?
   - Experiments: Compare to estimated-depth baseline

3. **Quality vs Speed Tradeoff**
   - Research question: What's the minimal ControlNet enhancement for acceptable quality?
   - Experiments: Vary inference steps, measure perceptual quality

### Phase 3: Contributions to Write
- Novel pipeline architecture diagram
- Quantitative evaluation on physics accuracy
- User study on perceived realism
- Ablation study (with/without physics, with/without ControlNet)
