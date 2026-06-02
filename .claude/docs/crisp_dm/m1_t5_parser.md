# CRISP-DM: M1 -- T5 Scene Parser

## 1. Business Understanding

**Question**: Can a natural language prompt be converted into a structured scene description?

**Goal**: Parse sentences like *"a person nervously walks toward a red box and kicks it"*
into a machine-readable JSON object:

```json
{
  "entities": [{"name": "person", "object_type": "humanoid"}, {"name": "box", "color": "red"}],
  "actions": [{"actor": "person", "action_type": "walk", "modifier": "nervously", "order": 0},
               {"actor": "person", "action_type": "kick", "order": 1}]
}
```

**Do we need ML?** Yes -- rule-based parsers cannot handle the full range of
natural language variation (modifiers, implied actors, co-reference, synonyms).

**Success metric**: Entity-F1 >= 0.80, Action-F1 >= 0.75 on a held-out validation set.
JSON syntax parse rate >= 95%.

---

## 2. Data Understanding

| Dataset | Size | Content | Quality |
|---------|------|---------|---------|
| **Synthetic (generated)** | ~5k pairs | Script-generated prompts + JSON | High (deterministic) |
| **Inter-X text captions** | ~1k | Human-written interaction captions | Medium (short, action-focused) |
| **AMASS filenames** | ~500 | Action verbs from sequence names | Low (sparse labels) |

**Gaps**:
- No dataset with complex multi-actor, multi-action, multi-modifier prompts
- AMASS filenames have no sentence structure -- cannot use directly
- Inter-X captions focus on actions, rarely describe scene objects

**Trade-off accepted**: Synthetic data dominates training. Model may overfit to the
generation template. Mitigated by synonym expansion in the generator.

---

## 3. Data Preparation

**Architecture**: Flan-T5-Small (60M params) trained as a seq2seq model.
- **Input**: Natural language prompt (tokenized by T5 tokenizer)
- **Output**: JSON string using sentinel-token brace replacement
  (`<extra_id_0>` -> `{`, `<extra_id_1>` -> `}`) to avoid T5's trouble with
  raw curly braces in the generation target

**Why sentinel tokens?** T5 uses `{}`-like tokens in its span-corruption pre-training;
injecting them directly as generation targets confuses the model. Replacing `{}` with
`<extra_id_N>` exploits tokens the model already understands structurally.

**Splits**: 80/10/10 train/val/test.

---

## 4. Modelling

**Architecture choice**: Flan-T5-Small over alternatives:
- GPT-2 (decoder-only): Cannot leverage the structured sentinel tokens
- T5-Base: 220M params -- too large for a 6GB VRAM GPU (RTX 3050)
- Rule-based parser: Cannot handle synonym, modifier, or co-reference variation

**Training settings** (from `checkpoints/understanding/scene_extractor_v5`):
- Optimizer: AdamW, lr=5e-5, epochs=30
- Loss: standard T5 cross-entropy (seq2seq)
- `predict_with_generate=False` during training (evaluation speed)
- `predict_with_generate=True` for final evaluation (required for BLEU/ROUGE)

**Current checkpoint**: v5 (293 MB). Training ran locally on Windows RTX 3050.

---

## 5. Evaluation

| Metric | Current | Target | Notes |
|--------|---------|--------|-------|
| JSON syntax rate | Unknown* | >= 95% | *Disabled during training |
| Entity-F1 | Unknown* | >= 0.80 | Measured on val set |
| Action-F1 | Unknown* | >= 0.75 | Measured on val set |

> [!WARNING]
> `predict_with_generate=False` means BLEU/ROUGE/F1 were never computed during training.
> **Action**: Re-run evaluation with `predict_with_generate=True` to measure actual performance.

---

## 6. Deployment

- Loaded via `T5SceneParser(device="cuda", fallback=False)` in `UnderstandingStage`
- Fallback to `PromptParser` (rule-based) if T5 checkpoint not found
- Serving: local GPU inference, ~50ms per prompt on RTX 3050

**Next improvement**: Fine-tune on Inter-X + PAHOI text annotations to improve
multi-person interaction parsing.
