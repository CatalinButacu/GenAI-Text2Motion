from __future__ import annotations

from text2motion.studio.avatar import SmplxAvatarController
from text2motion.studio.contracts import ViewerHost
from text2motion.studio.gui.avatar_section import AvatarSection
from text2motion.studio.gui.generation_panel import GenerationPanel
from text2motion.studio.gui.layout import PanelLayout
from text2motion.studio.gui.playback import PlaybackPanel
from text2motion.studio.gui.prompt import PromptPanel
from text2motion.studio.gui.sampling import SamplingSection
from text2motion.studio.gui.scene_panels import EnvironmentPanel, ScenePanel
from text2motion.studio.live.controller import StudioGenerationController


############################################################
#       GuiPanels
############################################################
class GuiPanels:
    def __init__(
        self,
        host: ViewerHost,
        state,
        avatar: SmplxAvatarController,
        generation: StudioGenerationController,
    ) -> None:
        self.host = host
        self.state = state
        self.avatar = avatar
        self.generation = generation
        layout = PanelLayout(state)
        self.scene = ScenePanel(host, layout)
        self.environment = EnvironmentPanel(host, layout)
        self.params = GenerationPanel(
            host,
            state,
            generation,
            layout,
            SamplingSection(host, state, layout),
            AvatarSection(state, avatar, layout),
        )
        self.prompt = PromptPanel(state, generation, layout)
        self.playback = PlaybackPanel(host, state, layout)

    def install(self) -> None:
        self.host.gui_controls["environment"] = self.environment.gui_environment
        self.host.gui_controls["params"] = self.params.gui_params
        self.host.gui_controls["prompt"] = self.prompt.gui_prompt

    def gui_scene(self) -> None:
        self.scene.gui_scene()

    def gui_playback(self) -> None:
        self.playback.gui_playback()
