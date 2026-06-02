# Datasets

All datasets used in the project -- sources, formats, and status.

---

## Dataset Inventory

| # | Dataset | Module | Size | Status |
|---|---------|--------|------|--------|
| D1 | **KIT-ML** | M4 (motion) | 4,886 train / 300 val |  Ready |
| D2 | **M1 Training Set** | M1 (parser) | 40,000 JSONL (35.5 MB) |  Built |
| D3 | **Visual Genome** | M1 (source data) | 4.85 GB |  Downloaded |
| D4 | **Physics State** | M4 (PhysicsSSM) | Derived from D1 |  Built |
| D5 | **KB Index** | M2 (retriever) | 12 curated objects |  Built |
| D6 | **Inter-X** | Multi-person SMPL-X | 11,388 samples (~40 GB) |  Downloaded |
| D7 | **PAHOI** | Human-object interactions | 562 sequences |  Available |
| D8 | **HumanML3D** | Future upgrade for M4 | 14,616 samples |  Not yet used |

---

## D1: KIT-ML Motion Features (251 dimensions)

| Index | Content | Dims | Description |
|-------|---------|------|-------------|
| 0:3 | Root angular velocity | 3 | Pelvis rotation (rad/s) |
| 3:6 | Root linear velocity | 3 | Pelvis translation (m/s) |
| 9:63 | Joint rotations (6D) | 54 | 9 joints x 6D continuous rotation |
| 63:66 | Root joint position | 3 | World-space XYZ |
| 66:129 | Joint positions | 63 | 21 joints x 3 relative to root |
| 129:192 | Joint velocities | 63 | 21 joints x 3 (m/s) |
| 192:251 | Foot contacts + extras | 59 | Binary contacts + features |

---

## D6: Inter-X (SMPL-X, 168 dimensions)

| Field | Dims | Description |
|-------|------|-------------|
| global_orient | 3 | Root orientation (axis-angle) |
| body_pose | 63 | 21 body joints x 3 |
| left_hand_pose | 45 | 15 hand joints x 3 |
| right_hand_pose | 45 | 15 hand joints x 3 |
| transl | 3 | Global translation |
| betas | 10 | Body shape parameters |

---

## File Locations

```
data/
+ KIT-ML/              # D1: motion feature vectors (.npy)
+ m1_training/         # D2: T5 training JSONL
+ M1_VisualGenome/     # D3: raw VG images + annotations
+ knowledge_base/      # D5: KB index + embeddings
+ inter-x/             # D6: Inter-X SMPL-X sequences
+ training/            # Preprocessed training data
```
