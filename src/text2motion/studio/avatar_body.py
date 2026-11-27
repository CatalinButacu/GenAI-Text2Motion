from __future__ import annotations

from dataclasses import replace

import numpy as np
import torch

from text2motion.studio.contracts import SmplxFitResult
from text2motion.studio.fitting import build_smplx_body_model
from text2motion.studio.pose_parameters import upsample_pose_parameters


############################################################
#       SmplxBodyModels
############################################################
class SmplxBodyModels:
    def __init__(self, state) -> None:
        self.state = state

    def body_model(self, frames: int, gender: str):
        model = self.state.smplx_fit.model_cache.get((frames, gender))
        if model is None:
            cfg = replace(self.state.config.fit, model_dir=self.state.model_dir, gender=gender)
            model = build_smplx_body_model(cfg, frames, self.state.device)
            self.state.smplx_fit.model_cache[(frames, gender)] = model
        return model

    def upsample_fit(self, res: SmplxFitResult, factor: int, gender: str) -> SmplxFitResult:
        if factor == 1:
            return res
        buffers = self.state.stream_buffers
        orient, pose, transl = upsample_pose_parameters(
            buffers.fit_anchor, res.global_orient, res.body_pose, res.transl, factor
        )
        buffers.fit_anchor = (res.global_orient[-1], res.body_pose[-1], res.transl[-1])
        model = self.body_model(len(orient), gender)
        dev = self.state.device if torch.cuda.is_available() else "cpu"
        with torch.no_grad():
            out = model(
                global_orient=torch.tensor(orient, device=dev),
                body_pose=torch.tensor(pose, device=dev),
                transl=torch.tensor(transl, device=dev),
                betas=torch.tensor(res.betas, dtype=torch.float32, device=dev)
                .unsqueeze(0)
                .expand(len(orient), -1),
            )
        return replace(
            res,
            vertices=out.vertices.cpu().numpy(),
            global_orient=orient,
            body_pose=pose,
            transl=transl,
        )

    def shaped_vertices(self, model, res, t: int) -> np.ndarray:
        user = self.state.user_betas
        if not user.any():
            return res.vertices
        dev = self.state.device if torch.cuda.is_available() else "cpu"
        with torch.no_grad():
            out = model(
                global_orient=torch.tensor(res.global_orient, device=dev),
                body_pose=torch.tensor(res.body_pose, device=dev),
                transl=torch.tensor(res.transl, device=dev),
                betas=torch.tensor(res.betas + user, dtype=torch.float32, device=dev)
                .unsqueeze(0)
                .expand(t, -1),
            )
        return out.vertices.cpu().numpy()
