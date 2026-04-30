# Physics Engine - Deep Research Analysis

## The Giants We Stand On

### 1. Bullet Physics / PyBullet (Coumans, 2015)
**Source**: [pybullet.org](https://pybullet.org) | [Quickstart Guide](https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA)

#### What They Solved
- Real-time rigid body dynamics (gravity, collisions, friction)
- Python API for robotics/ML research
- URDF/SDF model loading, RGBD camera rendering

#### What We Borrow
| Concept | How We Use It | Our Configuration |
|---------|--------------|-------------------|
| `stepSimulation()` | Advance physics by dt | 1/240s timestep (default) |
| `getCameraImage()` | Render RGB + Depth | 640x480, used for ControlNet |
| `createCollisionShape()` | Define collision geometry | Box, sphere, mesh |
| `applyExternalForce()` | Implement actions | Force from Story Agent |

#### What PyBullet Provides That Others Don't
| Feature | MuJoCo | PyBullet | Our Benefit |
|---------|--------|----------|-------------|
| License | Recently free | Always free | Easy distribution |
| Mesh loading | Limited | Full OBJ support | Shap-E integration |
| Python API | Wrapper | Native | Simpler code |
| Camera render | Basic | Full RGBD | ControlNet input |

---

### 2. OpenAI Gym + PyBullet Environments
**Link**: [pybullet-gym](https://github.com/bulletphysics/bullet3/tree/master/examples/pybullet/gym)

#### What They Solved
- Standardized RL environment interface
- Pre-built robot models and tasks

#### What We Learn From It
- How to structure simulation loops
- Best practices for camera positioning
- Action space design patterns

---

## YOUR ORIGINAL CONTRIBUTION

### The Research Gap We Fill
> **Current state**: PyBullet used for RL/robotics with URDF robots
> **Our contribution**: PyBullet as a text-to-video rendering engine

### What Makes Your Work Novel

1. **Cinematic Camera System**
   - PyBullet default: static camera
   - **Ours**: Orbit, zoom, pan, pitch with easing functions
   - Parsed from natural language ("camera orbits around the scene")

2. **Action-Time Scheduling**
   - PyBullet default: forces applied immediately
   - **Ours**: Scheduled actions at specific times
   ```python
   actions = [
       {"time": 0.5, "object": "ball", "type": "force", "force": [0, 0, 10]}
   ]
   ```

3. **Dual Output: Physics + AI**
   - RGB frames -> "sim graphics" output
   - Depth frames -> ControlNet input for realistic output

### The Physics Guarantee
> **Key insight**: If it moves correctly in PyBullet, it's physically correct.
> This is our "ground truth" that AI-only generators cannot provide.

### Where You Investigate Further
- [ ] **Soft Body Physics**: Cloth, liquids (pybullet has experimental support)
- [ ] **Multi-Object Collision Chains**: "dominoes falling"
- [ ] **Camera Tracking**: Follow specific object through scene
- [ ] **Slow Motion**: Variable simulation speed for dramatic effect
