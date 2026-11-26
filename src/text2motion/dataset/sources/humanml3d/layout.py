from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from text2motion.motion.storage import FEATURES_DIR


############################################################
#       Hml3dPreparationLayout
############################################################
@dataclass(frozen=True)
class Hml3dPreparationLayout:
    out_dir: Path

    @property
    def pose_data(self) -> Path:
        return self.out_dir / "pose_data"

    @property
    def joints(self) -> Path:
        return self.out_dir / "joints"

    @property
    def new_joints(self) -> Path:
        return self.out_dir / "new_joints"

    @property
    def new_joint_vecs(self) -> Path:
        return self.out_dir / FEATURES_DIR

    def mkdirs(self) -> None:
        for d in (self.pose_data, self.joints, self.new_joints, self.new_joint_vecs):
            d.mkdir(parents=True, exist_ok=True)
