# Story Agent - Deep Research Analysis

> **NOTE**: This document is **research background / design inspiration** for M1 (Scene Parser) and M2
> (Knowledge Retriever). There is **no `StoryAgent` class or `story_agent.py`** in the codebase.
> The concepts described here are incorporated directly into M1+M2.

## The Giants We Stand On

### 1. Text2Scene (CVPR 2019) - Compositional Scene Generation
**Paper**: Tan et al., "Text2Scene: Generating Compositional Scenes from Textual Descriptions"

#### What They Solved
- Generated 2D scene layouts from text without GANs
- Used sequential object placement based on semantic parsing
- Addressed spatial relationships ("on", "next to", "behind")

#### What We Borrow
| Concept | How We Use It | Our Improvement |
|---------|--------------|-----------------|
| Entity extraction | Regex patterns for "ball", "cube", "table" | Add spaCy NLP for verb-object detection |
| Spatial prepositions | Map "on" -> position offset | Extend to 3D coordinates |
| Sequential generation | Process objects one-by-one | Maintain for physics ordering |

#### Gap They Left
- **Only 2D** - no depth/height handling
- **No physics** - objects just placed, not simulated
- **Static scenes** - no motion or dynamics

---

### 2. Stanford Spatial Common Sense (EMNLP 2018)
**Paper**: Chang et al., "Learning Spatial Common Sense with Geometry-Aware Recurrent Networks"

#### What They Solved
- Inferred implicit spatial constraints from text
- "A computer on a desk" -> system knows desk is below, supporting
- Built lookup tables for preposition-to-relation mapping

#### What We Borrow
| Concept | How We Use It | Our Improvement |
|---------|--------------|-----------------|
| Support relationships | "on" means Y < Z axis | Add physics constraint (static vs dynamic) |
| Default heights | Objects placed at realistic heights | Auto-assign mass based on object type |
| Common sense inference | Unstated relationships | Use LLM fallback for ambiguous cases |

#### Gap They Left
- **No temporal reasoning** - no "falls", "bounces", "throws"
- **Scene understanding only** - no generation capability

---

## YOUR ORIGINAL CONTRIBUTION

### The Research Gap We Fill
> **Current state of art**: Text -> 2D/3D scene layouts OR Text -> Video (no physics)
> **Our contribution**: Text -> Physically-simulated 3D video with camera motion

### What Makes Your Work Novel

1. **Action Parsing for Physics**
   - Existing: "ball on table" -> static placement
   - **Yours**: "ball falls on table" -> `SceneAction(time=0, action_type="gravity")`

2. **Physics-Aware Scene Description**
   - Add `mass`, `is_static`, `velocity` to scene representation
   - Map natural language to physical properties

3. **Cinematic Integration**
   - Parse camera keywords ("zoom", "orbit", "pan")
   - Connect to `CinematicCamera` system

### Where You Investigate Further
- [ ] **Verb Classification**: Build taxonomy of verbs -> force types (fall=gravity, throw=impulse, push=force)
- [ ] **Physical Property Inference**: "heavy ball" -> mass=5.0, "light feather" -> mass=0.01
- [ ] **Temporal Language**: "then", "after", "while" -> action sequencing
