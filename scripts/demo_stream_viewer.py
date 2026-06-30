"""Interactive text-to-motion STUDIO: a 3D window with a prompt box -- type an action, the avatar
streams it live as a skeleton, then a fitted SMPL-X body mesh swaps in. Type another, repeat.

Ported from the donor's chat_viewer pattern (imgui input bar + background generation thread ->
lock-guarded pending clip -> on_update swaps the renderable every render frame so the GL loop stays
smooth). Our 263 track yields joint POSITIONS, so:
  - live feedback   = aitviewer Skeletons (streamed chunk-by-chunk, the bounded-memory novelty),
  - final body mesh = SMPL-X fit to those joints (joints2smpl; rot6d-direct distorts ~27 cm, so we
    optimise instead), shown as a plain Meshes node built from the fitted vertices (no SMPL model
    loaded in the viewer -> fits the 4 GB GPU alongside the generator).

    $env:PYTHONPATH="src"; .venv/Scripts/python.exe scripts/demo_stream_viewer.py

Type a prompt (Enter or Generate). The mesh fit takes ~30-60 s; pass --no-mesh for skeleton-only,
or --fit_device cpu if the GPU runs out of memory. Close the window to exit.
"""

from __future__ import annotations

import argparse
import threading
import time
from collections import deque
from dataclasses import replace
from pathlib import Path
from typing import Any

import imgui
import numpy as np
import torch
from aitviewer.renderables.meshes import Meshes
from aitviewer.renderables.plane import ChessboardPlane
from aitviewer.renderables.skeletons import Skeletons
from aitviewer.viewer import Viewer

from text2motion.model.generator import MotionGenerator
from text2motion.model.text_encoder import CLIPTextEncoder
from text2motion.model.tokenizer import ResidualFsqTokenizer
from text2motion.render.joints2smpl import FitConfig, fit_smplx_to_joints
from text2motion.render.studio import kinematic_bones, recover_skeleton
from text2motion.shared.config import load_config
from text2motion.shared.seed import seed_everything
from text2motion.stream.decode import StreamingMotionDecoder

MAX_INPUT_LEN = 256
MAX_HISTORY = 8
SKEL_COLOR = (0.20, 0.45, 0.85, 1.0)
MESH_COLOR = (0.55, 0.60, 0.90, 1.0)
STATUS_COLORS = {
    "ok": (0.5, 1.0, 0.5, 1.0),
    "running": (1.0, 0.85, 0.35, 1.0),
    "error": (1.0, 0.45, 0.45, 1.0),
}


def load_pipeline(a: argparse.Namespace, dev: str):
    cfg = load_config(a.config)
    tok = ResidualFsqTokenizer(cfg.tokenizer)
    tok.load_state_dict(torch.load(a.tokenizer_ckpt, map_location="cpu"))
    tok.to(dev).eval()
    nl = cfg.generator.mamba_n_layers if a.backbone == "mamba" else cfg.generator.n_layers
    gc = replace(
        cfg.generator,
        backbone=a.backbone,
        n_layers=nl,
        num_codebooks=cfg.tokenizer.num_quantizers,
        codebook_size=tok.codebook_size,
        use_kernel=False,
    )
    gen = MotionGenerator(gc).to(dev).eval()
    st = torch.load(a.ckpt, map_location="cpu")
    gen.load_state_dict(st["generator"] if isinstance(st, dict) and "generator" in st else st)
    te = CLIPTextEncoder(cfg.text_encoder).to(dev).eval()
    out = Path(cfg.paths.hml3d_out_dir)
    mean = torch.from_numpy(np.load(out / "Mean.npy").astype("float32")).to(dev)
    std = torch.from_numpy(np.load(out / "Std.npy").astype("float32")).to(dev)
    return tok, gen, te, mean, std


class StreamingStudioViewer(Viewer):
    """aitviewer window: prompt -> streamed Skeletons -> fitted SMPL-X Meshes (donor chat pattern)."""

    scene: Any  # aitviewer sets self.scene = None pre-construction; silence the type checker

    def __init__(self, a: argparse.Namespace, dev: str) -> None:
        super().__init__(title="Streaming Text-to-Motion  |  GenAI Text2Motion", size=(1280, 800))
        self.a = a
        self.dev = dev
        self.tok, self.gen, self.te, self.mean, self.std = load_pipeline(a, dev)
        self.bones = kinematic_bones()
        self.lock = threading.Lock()
        self.pending: np.ndarray | None = None  # accumulated skeleton (T, 22, 3), lock-guarded
        self.pending_reset = False  # first chunk of a new prompt -> restart playback at frame 0
        self.pending_mesh: tuple[np.ndarray, np.ndarray] | None = None  # (verts, faces)
        self.busy = False
        self.stage = ""  # "generating" / "fitting mesh" -> shown in the chat panel
        self.input_buffer = a.prompt
        self.history: deque[tuple[str, str]] = deque(maxlen=MAX_HISTORY)

        self.scene.fps = a.fps
        self.playback_fps = a.fps
        self.scene.background_color = (0.92, 0.93, 0.95, 1.0)
        self.run_animations = True
        if self.scene.floor is not None:
            self.scene.remove(self.scene.floor)
        floor = ChessboardPlane(100.0, 200, (0.82, 0.83, 0.84, 1.0), (0.80, 0.81, 0.82, 1.0), "xz")
        self.scene.floor = floor
        self.scene.add(floor)
        cam = self.scene.camera
        if cam is not None:
            cam.position = np.array([2.0, 1.8, 4.5])
            cam.target = np.array([0.0, 1.0, 0.0])

        self.gui_controls["chat"] = self.gui_chat
        if a.prompt:
            self.submit_prompt(a.prompt)  # auto-run the initial prompt on open

    def submit_prompt(self, prompt: str) -> None:
        prompt = prompt.strip()
        if not prompt or self.busy:
            return
        self.busy = True
        self.history.append((prompt, "running"))
        threading.Thread(target=self.producer, args=(prompt,), daemon=True).start()

    def producer(self, prompt: str) -> None:
        """Stream skeleton chunk-by-chunk (live), then fit a SMPL-X mesh to the joints."""
        a = self.a
        status = "ok"
        self.stage = "generating"
        with torch.no_grad():
            emb = self.te([prompt]).to(self.dev)
            dec = StreamingMotionDecoder(self.tok, self.mean, self.std)
            it = self.gen.stream(
                emb,
                a.steps,
                temperature=a.temperature,
                top_p=a.top_p,
                cfg_scale=a.cfg_scale,
                stop_at_end=not a.fixed_length,
            )
            acc: np.ndarray | None = None
            for chunk in dec.stream_tokens(it):
                joints = recover_skeleton(chunk.squeeze(0).cpu().numpy())  # (cf, 22, 3)
                acc = joints if acc is None else np.concatenate([acc, joints], axis=0)
                with self.lock:
                    self.pending = acc.copy()
                    self.pending_reset = acc.shape[0] == joints.shape[0]  # true on the first chunk
                if a.stream_fps > 0:
                    time.sleep(joints.shape[0] / a.stream_fps)  # throttle to make growth visible

        if acc is None:
            status = "no motion"
        elif a.mesh:
            self.stage = "fitting mesh"
            if self.dev == "cuda":
                torch.cuda.empty_cache()  # release streaming activations before the fit
            cfg = FitConfig(model_dir=a.model_dir, stage2_iters=a.fit_iters)
            res = fit_smplx_to_joints(acc, cfg, device=a.fit_device)
            with self.lock:
                self.pending_mesh = (res.vertices, res.faces)
            status = f"ok ({res.joint_err_cm:.1f} cm)"

        self.stage = ""
        self.busy = False
        if self.history and self.history[-1][0] == prompt:
            self.history[-1] = (prompt, status)

    def _clear_motion_nodes(self) -> None:
        for node in [n for n in self.scene.nodes if isinstance(n, (Skeletons, Meshes))]:
            self.scene.remove(node)

    def on_update(self) -> None:
        with self.lock:
            mesh = self.pending_mesh
            joints, reset = self.pending, self.pending_reset
            self.pending_mesh = self.pending = None
        if mesh is not None:  # mesh wins: it is the final, higher-fidelity view
            verts, faces = mesh
            self._clear_motion_nodes()
            body = Meshes(verts, faces, color=MESH_COLOR, flat_shading=False)
            body.name = "body"
            self.scene.add(body)
            self.scene.current_frame_id = 0
            self.run_animations = True
            return
        if joints is None:
            return
        self._clear_motion_nodes()
        seq = Skeletons(joint_positions=joints, joint_connections=self.bones, color=SKEL_COLOR)
        seq.name = "stream"
        self.scene.add(seq)
        if reset:
            self.scene.current_frame_id = 0
        self.run_animations = True

    def gui_chat(self) -> None:
        w, h = self.window_size
        cw = min(720, w - 40)
        imgui.set_next_window_position((w - cw) / 2, h - 160, imgui.ALWAYS)
        imgui.set_next_window_size(cw, 150, imgui.ALWAYS)
        imgui.set_next_window_bg_alpha(0.9)
        opened, _ = imgui.begin("Text-to-Motion", None, imgui.WINDOW_NO_COLLAPSE)
        if not opened:
            imgui.end()
            return

        imgui.begin_child("##hist", height=58, border=False)
        for prompt, status in self.history:
            imgui.text_colored(
                f"> {prompt}", *STATUS_COLORS.get(status[:2], STATUS_COLORS["error"])
            )
            if status not in ("ok", "running"):
                imgui.same_line()
                imgui.text_disabled(f"[{status}]")
        imgui.set_scroll_here_y(1.0)
        imgui.end_child()
        if self.stage:
            imgui.text_colored(f"... {self.stage} ...", 1.0, 0.85, 0.35, 1.0)
        imgui.separator()

        imgui.set_next_item_width(cw - 130)
        enter, self.input_buffer = imgui.input_text(
            "##prompt", self.input_buffer, MAX_INPUT_LEN, imgui.INPUT_TEXT_ENTER_RETURNS_TRUE
        )
        imgui.same_line()
        if self.busy:
            imgui.push_style_var(imgui.STYLE_ALPHA, 0.3)
            imgui.button("Generate", width=110)
            imgui.pop_style_var()
            send = False
        else:
            send = imgui.button("Generate", width=110)
        if (enter or send) and self.input_buffer.strip() and not self.busy:
            self.submit_prompt(self.input_buffer)
        imgui.end()

    def gui(self) -> None:
        self.on_update()
        super().gui()


def main() -> None:
    p = argparse.ArgumentParser(
        description="Interactive streaming text-to-motion studio (aitviewer)."
    )
    p.add_argument("--config", default="configs/generator/final100m_fsq8x1024.yaml")
    p.add_argument("--backbone", default="transformer", choices=["transformer", "mamba"])
    p.add_argument("--ckpt", default="checkpoints/generator/generator_transformer.pt")
    p.add_argument("--tokenizer_ckpt", default="checkpoints/tokenizer/fsq_g8_v1024.pt")
    p.add_argument(
        "--prompt", default="a person walks forward and waves", help="prompt run on open"
    )
    p.add_argument("--steps", type=int, default=49)
    p.add_argument("--cfg_scale", type=float, default=6.0)  # locked val-sweep optimum
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--fps", type=int, default=20)  # our motion rate
    p.add_argument(
        "--stream_fps",
        type=int,
        default=20,
        help="throttle skeleton growth to N fps; 0 = full speed",
    )
    p.add_argument("--fixed_length", action="store_true")
    p.add_argument(
        "--mesh",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="fit + show a SMPL-X body mesh after streaming (--no-mesh = skeleton only)",
    )
    p.add_argument(
        "--model_dir",
        default=r"D:\Facultate\dissertation\data\arctic\unpack\models",
        help="dir containing smplx/SMPLX_NEUTRAL.npz",
    )
    p.add_argument(
        "--fit_device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="device for the SMPL-X fit; use cpu if the 4 GB GPU OOMs",
    )
    p.add_argument("--fit_iters", type=int, default=300)
    a = p.parse_args()
    seed_everything(2026, False)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    StreamingStudioViewer(a, dev).run()


if __name__ == "__main__":
    main()
