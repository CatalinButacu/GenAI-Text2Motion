# Model Generator - Deep Research Analysis

## The Giants We Stand On

### 1. Shap-E (OpenAI, 2023) - Text to 3D Implicit Functions
**Paper**: Jun & Nichol, "Shap-E: Generating Conditional 3D Implicit Functions"
**Link**: [arXiv:2305.02463](https://arxiv.org/abs/2305.02463)

#### What They Solved
- Text/image -> 3D mesh in seconds (not hours like DreamFusion)
- Generates NeRF + textured mesh parameters directly
- Open-source, runs on consumer GPUs

#### What We Borrow
| Concept | How We Use It | Our Improvement |
|---------|--------------|-----------------|
| Text-to-mesh pipeline | `ShapEPipeline.from_pretrained()` | Wrap in retry logic |
| OBJ/PLY export | `export_to_obj()` utility | Auto-scale for PyBullet |
| Guidance scale param | Control generation fidelity | Tune for physics-ready meshes |

#### Gap They Left (Critical!)
> **Shap-E produces visual meshes, NOT physics-ready models**

| Missing | Why It Matters | Our Solution |
|---------|----------------|--------------|
| Mass | Needed for dynamics | Estimate from volume + material prompt |
| Friction | Affects collisions | Default values per object type |
| Collision shape | Complex mesh = slow sim | Compute convex hull for physics |
| Scale | Arbitrary output size | Normalize to real-world dimensions |

---

### 2. TripoSR (StabilityAI, 2024) - Fast Image-to-3D
**Paper**: VAST-AI Research, GitHub

#### What They Solved
- Single image -> 3D mesh in <0.5s
- Better for real objects (photos)
- Higher detail than Shap-E for some cases

#### What We Borrow
| Concept | How We Use It |
|---------|--------------|
| Hybrid pipeline | SD image -> TripoSR -> mesh |
| Speed | Fallback when Shap-E fails |

---

### 3. DreamFusion (Google, 2022) - Score Distillation
**Paper**: Poole et al., "DreamFusion: Text-to-3D using 2D Diffusion"

#### What They Solved
- High quality text -> 3D via NeRF optimization
- Uses 2D diffusion as 3D prior

#### Why We DON'T Use It Directly
- **Too slow**: 1-2 hours per object
- **Not real-time**: Defeats video generation purpose

#### What We Learn From It
- Quality benchmark - our Shap-E outputs should approach this quality eventually

---

## YOUR ORIGINAL CONTRIBUTION

### The Research Gap We Fill
> **Current state**: Text-to-3D produces "pretty" meshes with NO physical properties
> **Our contribution**: Bridge text-to-3D with physics simulation

### What Makes Your Work Novel

1. **Physics Property Inference Layer**
```
"a wooden ball" -> {
    mesh: Shap-E("wooden ball"),
    mass: estimate_mass(volume, density["wood"]),
    friction: 0.5,
    restitution: 0.3  # bounciness
}
```

2. **Mesh-to-Collision Pipeline**
   - Generate mesh -> Compute bounding box -> Create PyBullet collision shape
   - Handle non-convex meshes via decomposition

3. **Graceful Fallback Chain**
   - Try Shap-E -> Fail? -> Try primitive (sphere/box) -> Log for improvement

### Where You Investigate Further
- [ ] **Volume Estimation**: Compute mesh volume for mass calculation
- [ ] **Material Recognition**: Parse "wooden", "metal", "glass" -> physical properties table
- [ ] **Mesh Simplification**: Reduce complexity for faster physics while keeping appearance
