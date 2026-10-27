# tokenization

This package turns motion into tokens and tokens back into motion. The motion is a clip of
HumanML3D-263 features, one 263-value row per frame. Both datasets of the project use this
format: the HumanML3D clips, and the AMASS clips after their conversion to 263 features. Both
are normalized with the HumanML3D train-split mean and standard deviation (`MotionScaler`),
because the AMASS folder has no statistics of its own. A token step covers `downsample` frames
(4 by default), and each step holds one code index for each codebook. The generator learns on
these token indices, so the work in this package comes first: you configure a tokenizer, train
it, measure it, freeze it, and then encode the corpus with it.

## Which data goes where

Only HumanML3D trains the tokenizer. The training loader reads the train split of
`hml3d_out_dir`, and each evaluation reads its validation split. AMASS has no split files, so no
training or evaluation reads it. AMASS goes through the tokenizer only after training, when
`build_token_pack` encodes it for generator pretraining. Because both datasets use the same mean
and standard deviation, one motion gives the same codes from either dataset. The YAML field
`data.sources` does not change this, because no code reads it.

The reasons for this choice:

- The tokenizer defines the vocabulary of the generator. A tokenizer trained on other data gives
  other codes, and both twin generators must then train again.
- The reconstruction FID that you can compare with published results exists only for HumanML3D.
  AMASS has no comparable score.
- HumanML3D comes from AMASS, so AMASS adds mostly motion of the same type to the tokenizer.
  The generator gets the AMASS variety in its pretraining.

A tokenizer pretrained on AMASS and then fine-tuned on HumanML3D is the planned experiment E7
(lesson 18). Nobody has run it. The reconstruction error of the current tokenizer on AMASS clips
is not measured.

## Configure

You start with a `TokenizerConfig`. It describes the main tokenizer, a finite scalar quantizer
(FSQ) that needs no learned codebook. The YAML section `tokenizer:` fills it. Its `kind` is a
`TokenizerKind`: `FINITE_SCALAR` (`fsq` on the command line) or `RESIDUAL_VECTOR` (`rvq`). When
you select `rvq`, the baseline with learned codebooks, its settings come from a second config,
`ResidualVectorQuantizationConfig`, which the YAML section `rvq_baseline:` fills.

## Build

With the configs ready, `build_tokenizer_network(finite_scalar_config, residual_vector_config)`
gives you a `TokenizerNetwork`. The `kind` in the first config selects the network, and the second
config matters only for `rvq`. The network is a normal PyTorch module, so you move it to a device
and load a state dict as usual.

A `TokenizerNetwork` has three calls. `encode(motion)` takes a float tensor of shape
`[batch, frames, 263]` and returns integer indices of shape `[batch, frames / downsample,
codebooks]`. `decode(indices)` does the opposite and returns features of shape `[batch, frames,
263]`. `forward(motion)` does both in one pass, for training, and returns a
`TokenizerReconstruction` with the reconstruction, the token indices and the commitment loss. The
commitment loss is zero for FSQ. Give a frame count that is a multiple of `downsample`. Other
counts do not raise an error, but the output is shorter: 62 frames in give 60 frames out.

## Train

To train a new network, wrap it in a `TokenizerTrainer(network, lr, weight_decay, ema_decay,
commitment_beta)`. Use `commitment_beta` only for `rvq`. Each `train_step(motion)` takes one
batch of `[batch, window, 263]` features, does one AdamW step and returns a dict of floats:
`recon`, `commit`, `total`, `perplexity` and `usage_frac`. The trainer also keeps an exponential
moving average (EMA) of the weights. The evaluations and the saved checkpoint use these averaged
weights, not the raw ones.

You do not usually call `train_step` yourself. `train_tokenizer(trainer, loader, request, device,
evaluate, checkpoint_path, resume_path)` runs the full loop for the epochs in a
`TrainTokenizerRequest`. Every `eval_every` epochs, and at the last epoch, it calls
`evaluate(Split.VALIDATION)` with the EMA weights in place. Your `evaluate` function must return a
dict with `recon_fid`, `mpjpe_mm`, `feature_l2` and `clips`. When `recon_fid` improves, the loop
saves the EMA weights to `checkpoint_path`. At each evaluation it also saves the full training
state to `resume_path`, so a run that stops can continue with `resume=True` from the last
evaluation. It returns the best validation reconstruction FID.

`ExponentialMovingAverage(model, decay)` is the EMA tool on its own. Call `update(model)` after
each optimizer step, `copy_to(model)` before an evaluation and `restore(model)` after it.

## Evaluate

To measure a network, call `evaluate_tokenizer_reconstruction(network, root, scaler, max_frames,
downsample, device, max_clips, split, reconstruction_fid_scorer)`. It encodes and decodes the clips
of one split from the dataset folder `root` and compares the result with the original clips. Only
the HumanML3D folder has splits, so `root` is the HumanML3D folder, and AMASS clips are not
measured with this function. It
returns `clips`, `mpjpe_mm` (the mean joint position error in millimetres) and `feature_l2`. When
you give it a `reconstruction_fid_scorer`, it also returns `recon_fid`. A lower value is better for
all three measures. Use the test split only once, after training, on the best checkpoint.

## Freeze and use

After training, the network does not change again. `FrozenMotionTokenizer(network, downsample,
device)` puts it in evaluation mode and works with clips in place of tensors.
`encode(clip)` takes a `MotionClip` with normalized features, cuts it to a multiple of
`downsample` frames and returns `MotionTokens` with indices of shape `[steps, codebooks]` and the
kept frame count. A clip shorter than one token step raises a `ValueError`. `decode(tokens)` gives
a `MotionClip` back. The application reads `codebook_size` from this object to size the
generator.

## Tokenize the corpus

The last step encodes the AMASS pretraining corpus once (`data/AMASS_263/new_joint_vecs` by
default), so the generator never runs the tokenizer during its own pretraining. The HumanML3D
clips get their tokens later, when the generator fine-tunes on them. `build_token_pack(tokenizer,
scaler, request, excluded, identity)` takes the frozen tokenizer, the HumanML3D scaler and a
`TokenizeRequest`. The request gives the features folder, the window length and stride, and the
output file. The function cuts each clip into windows, skips the clip names in `excluded` (each
AMASS clip that is the source of a HumanML3D validation or test clip), encodes each window and
writes one `.npz` file. In that file each window is an `int16` array of shape `[steps,
codebooks]`, and one extra entry holds a JSON identity record. The record contains your `identity`
dict, the source count and checksum, and the window settings. The function returns a short
summary: `sequences`, `excluded`, `segments`, `token_steps` and `pack_mb`. The default output
file, `data/amass_tokens.npz`, is not the file that generator pretraining reads by default
(`data/amass_tokens_fsq8x1024.npz`). Set `token_pack` to the file that pretraining reads.

To read a pack later, `read_pack_identity(path)` returns the identity record as a dict, so you can
check that the pack matches the tokenizer you load. It raises a `ValueError` for a file without a
record. `pack_clip_keys(pack)` takes the open `np.load` result and returns the window keys without
the record entry.

Every other name in the package is internal. Use the names above, and import each one from its
file, for example `from text2motion.tokenization.nn_model import FrozenMotionTokenizer`.
