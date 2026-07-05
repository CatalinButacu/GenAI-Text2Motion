from pathlib import Path

import numpy as np
import torch

TRANS_MATRIX = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
    ]
)
EX_FPS = 20  # target frame rate (official ``ex_fps``)
NUM_BETAS = 16  # SMPL-X AMASS stores 16 shape coefficients
NUM_JOINTS_OUT = 22  # HumanML3D body skeleton (SMPL-X joints[:22] == the SMPL body joints)
_GENDERS = ("male", "female", "neutral")


def _resolve_model_root(smplx_dir: Path) -> Path:
    if (smplx_dir / "smplx").is_dir():
        return smplx_dir

    if smplx_dir.name.lower() == "smplx":
        return smplx_dir.parent

    return smplx_dir


class AmassPoseExtractor:
    def __init__(
        self,
        smplx_dir: Path,
        num_betas: int = NUM_BETAS,
        num_joints_out: int = NUM_JOINTS_OUT,
        device: str = "cpu",
        chunk_frames: int = 256,
    ) -> None:
        import smplx  # lazy import: raises ImportError naturally if the package is absent

        self.device = torch.device(device)
        self.num_betas = num_betas
        self.num_joints_out = num_joints_out
        self.chunk_frames = chunk_frames
        model_root = _resolve_model_root(Path(smplx_dir))
        self._models: dict[str, object] = {}

        for gender in _GENDERS:
            model_file = model_root / "smplx" / f"SMPLX_{gender.upper()}.npz"
            if not model_file.is_file():
                raise FileNotFoundError(f"SMPL-X model not found: {model_file}")
            self._models[gender] = (
                smplx.create(
                    model_path=str(model_root),
                    model_type="smplx",
                    gender=gender,
                    use_pca=False,
                    flat_hand_mean=True,
                    num_betas=num_betas,
                )
                .to(self.device)
                .eval()
            )

        self._n_expr = int(self._models["neutral"].num_expression_coeffs)

    def _select_model(self, gender: object):
        g = gender.decode() if isinstance(gender, bytes) else str(gender)
        if g not in self._models:  # official: anything non-male/neutral -> female
            g = "female"
        return self._models[g]

    def _to_device(self, array: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(array, dtype=torch.float32, device=self.device)

    def _forward_chunk(
        self,
        bm: object,
        root: np.ndarray,
        body: np.ndarray,
        hand: np.ndarray,
        trans: np.ndarray,
        betas: np.ndarray,
        start: int,
        end: int,
    ) -> np.ndarray:
        n = end - start
        zeros3 = torch.zeros(n, 3, device=self.device)
        broadcast_betas = np.broadcast_to(betas, (n, betas.shape[0])).copy()
        body_out = bm(
            global_orient=self._to_device(root[start:end]),
            body_pose=self._to_device(body[start:end]),
            left_hand_pose=self._to_device(hand[start:end, :45]),
            right_hand_pose=self._to_device(hand[start:end, 45:]),
            jaw_pose=zeros3,
            leye_pose=zeros3,
            reye_pose=zeros3,
            expression=torch.zeros(n, self._n_expr, device=self.device),
            betas=self._to_device(broadcast_betas),
            transl=self._to_device(trans[start:end]),
        )
        return body_out.joints[:, : self.num_joints_out, :].detach().cpu().numpy()

    @torch.no_grad()
    def amass_to_pose(self, src_path: str | Path) -> np.ndarray | None:
        bdata = np.load(src_path, allow_pickle=True)

        if "trans" not in bdata:
            return None

        if "mocap_frame_rate" in bdata:  # SMPL-X release
            fps = float(bdata["mocap_frame_rate"])
        elif "mocap_framerate" in bdata:  # legacy SMPL-H release
            fps = float(bdata["mocap_framerate"])
        else:
            return None

        bm = self._select_model(bdata["gender"])
        down_sample = max(int(fps / EX_FPS), 1)
        sl = slice(None, None, down_sample)

        root = np.asarray(bdata["root_orient"][sl], dtype=np.float32)
        body = np.asarray(bdata["pose_body"][sl], dtype=np.float32)
        hand = np.asarray(bdata["pose_hand"][sl], dtype=np.float32)
        trans = np.asarray(bdata["trans"][sl], dtype=np.float32)
        betas = np.asarray(bdata["betas"][: self.num_betas], dtype=np.float32)
        total = root.shape[0]
        if total == 0:
            return None

        joint_chunks: list[np.ndarray] = []
        for start in range(0, total, self.chunk_frames):
            end = min(start + self.chunk_frames, total)
            chunk_joints = self._forward_chunk(bm, root, body, hand, trans, betas, start, end)
            joint_chunks.append(chunk_joints)

        joints = np.concatenate(joint_chunks, axis=0)
        return np.dot(joints, TRANS_MATRIX)
