---
description: Audit the HumanML3D training data and the verb label layer, then check other motion datasets for compatibility
agent: dataset-auditor
model: Gemini 4 Carbon (copilot)
---

Do 2 tasks. Report task 1 first.

## Task 1: reliability and academic rigor
1. Training data: check the HumanML3D clips and captions for problems that a reviewer would reject. Include label noise in the captions, duplicate or near-duplicate clips across train, validation and test (mirrored clips have the prefix `M`), caption time ranges, and known published critiques of HumanML3D.
2. Label layer: judge the 186 action groups as an annotation method.
   - Compute the agreement between the 2 label lists (status and group) with a standard measure, for example Cohen's kappa for status and the adjusted Rand index for groups. Say which library computes it.
   - Judge the tie-break with a third list, the decisions by the orchestrator in `scripts/qa/action_vocabulary.py` (`DECISIONS`), and the failed motion check.
   - Say what a thesis may claim with this layer, and what it may not claim.
3. Give a ranked list of at most 5 fixes. For each fix give its cost in hours and the claim that it protects.

## Task 2: more motion data
The owner wants to add data for rare actions. Check MotionMillion, Motion-X and Motion-X++, and any other public dataset that you find (for example BABEL, KIT-ML, HumanAct12 extensions).
For each dataset give:
- the license and how to get access;
- the motion format, skeleton, frame rate, and whether a conversion to HumanML3D-263 exists, with a link;
- the overlap with HumanML3D (the same AMASS clips count as a leak, not as new data);
- how many clips it adds for rare actions such as juggle, row, skate, ride, yawn;
- whether a frozen tokenizer can represent it, given the A14 result (AIST++ 3.67x, 100STYLE 7.43x the HumanML3D validation error, limit 2.0x);
- the effect on the comparison with published HumanML3D models;
- whether it fits before 2026-12-01 on one 4 GB GPU and 16 GB RAM.

End with one recommendation for task 2: add, add later as future work, or do not add, with the reason.
