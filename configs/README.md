# configs/ — layout

Sorted into subfolders by role (2026-06-29). All path references in scripts/src/infra/paper were
updated to match. Base configs stay at the root.

```
configs/
  default.yaml              base config (dataclass defaults; experiments inherit + override)
  aws.yaml                  cloud run config (relative paths, portable for the g5 launch)
  generator/               Contribution B
    gen_pilot_fsq8x1024.yaml    34M pilot (the local bs8 twin)
    final100m_fsq8x1024.yaml    100M final twin (transformer 98M / mamba 100M), the headline config
    final100m.yaml              earlier 100M (pre FSQ-8x1024 freeze)
    eval_pilot31m.yaml          31M evaluation pilot
  tokenizer/               Contribution A
    fsq_g8_v1024.yaml           the frozen winner (recon-FID 0.0170); feeds every generator run
    fsq_g{4,6,8}_v{512,1024}.yaml   FSQ grid sweep
    rvq_l{4,6,8}_v{512,1024}.yaml   strong-RVQ baseline sweep
    tokenizer_isovocab.yaml     iso-vocab 6x512 confound (test recon-FID 0.0871)
    tok_*.yaml                  older pre-rename configs (provenance)
```

Note: a config's internal `paths:` are relative to the repo root (cwd), not the file location, so
moving configs does not change how they resolve.
