# Data preprocessing (draft, state of 2026-11-11)

This section describes the steps that the code applies today. It does not describe the steps that are proposed and not applied: synthetic data, added noise, and class balancing. Those steps are not part of the reported results.

## Source and split

We use the HumanML3D dataset with its official split files. Each split lists clip identifiers. A clip and its mirrored copy share one base identifier.

| Split | Base clips | Listed entries (with mirror copies) |
|---|---|---|
| Train | 11 692 (80.0%) | 23 384 |
| Validation | 730 (5.0%) | 1 460 |
| Test | 2 192 (15.0%) | 4 384 |

Check: the base identifiers of the three splits share no clip. The check found zero overlap between train and validation, train and test, and validation and test.

## Mirror augmentation

- Training uses mirror augmentation. Each training clip is used with its x-mirrored copy.
- Validation and test do not use mirror augmentation. Evaluation loads the base identifiers only.

## Clip length

- Each clip keeps 40 to 196 frames.
- Clips shorter than 40 frames are dropped. In training, 966 clips are dropped by this rule.
- Evaluation drops generated clips when their length is outside the valid range. In one validation run, 34 clips were dropped this way, and 628 were scored.

## Normalization

The mean and the standard deviation of each feature come from the training clips only. Validation and test use these training statistics without change.

## Text

Each clip has several captions. Each caption is one text input. The text encoder is described in the model section. The current production generators use a CLIP text encoder. A change to another text encoder is under evaluation and is not applied to the reported results.

## Evaluation set

- Validation: 628 clips are scored. Another 34 clips were dropped during generation because their length was outside the valid range. The source of the set size (730 base validation clips) is not yet traced in the code.
- The reference motion of each clip is the real motion of that same clip.
- The generated motion comes from one of the captions of that clip.

## Checks

- Verb calibration: each verb is counted in the captions of each split. Shares and two-proportion z values are in a tracked table.
- 45 of 1 655 verb lemmas have at least 20 clips in each split. Only these verbs support a per-verb claim.

## Not applied

- No synthetic clips are added to any split.
- No noise is added to the features.
- No class balancing is applied to the training sampler.
- No split change is applied.
