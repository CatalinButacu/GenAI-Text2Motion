from __future__ import annotations

from pathlib import Path
from types import MappingProxyType

_FOLDER_NAMES = MappingProxyType(
    {
        "MPI_HDM05": "HDM05",
        "BioMotionLab_NTroje": "BMLrub",
        "MPI_Limits": "PosePrior",
        "MPI_mosh": "MoSh",
        "DFaust_67": "DFaust",
        "SSM_synced": "SSM",
        "TCD_handMocap": "TCDHands",
        "Transitions_mocap": "Transitions",
        "Eyes_Japan_Dataset": "EyesJapanDataset",
    }
)


_OFFICIAL_NAMES = MappingProxyType({folder: official for official, folder in _FOLDER_NAMES.items()})


DATASET_ALIASES = MappingProxyType(
    {
        "biomotionlab_ntroje": "bmlrub",
        "dfaust_67": "dfaust",
        "mpi_hdm05": "hdm05",
        "mpi_limits": "poseprior",
        "mpi_mosh": "mosh",
        "ssm_synced": "ssm",
        "transitions_mocap": "transitions",
        "eyesjapandataset": "eyes_japan_dataset",
    }
)


def dataset_name(source_path: str | Path) -> str:
    parts = [part for part in Path(source_path).parts if part not in (".", "pose_data")]
    if not parts:
        return ""
    return _OFFICIAL_NAMES.get(parts[0], parts[0])
