"""Standard HumanML3D-263 feature pipeline, faithfully ported from EricGuo5513/HumanML3D.

Modules:
  quaternion  -- quaternion / cont6d math (port of common/quaternion.py)
  skeleton    -- IK/FK skeleton (port of common/skeleton.py)
  param_util  -- exact SMPL/T2M constants (port of paramUtil.py + notebook __main__)
  feature     -- joints (T,22,3) -> 263 + recover_from_ric (port of motion_representation.ipynb)
  raw_pose    -- AMASS SMPL-H npz -> (T,22,3) Y-up joints (port of raw_pose_processing.ipynb)
  regenerate  -- end-to-end entry script (AMASS -> 263 + Mean/Std + mirror augmentation)
  dataset     -- torch Dataset over the regenerated 263 + Mean/Std + official splits + texts
"""
