from __future__ import annotations


class SceneNodeClearer:
    def __init__(self, scene, nodes) -> None:
        self.scene = scene
        self.nodes = nodes

    def drop(self, node):
        if node is None:
            return
        if self.scene.selected_object is node:
            self.scene.select(None)
        self.scene.remove(node)

    def clear_rest(self) -> None:
        self.nodes.rest = self.drop(self.nodes.rest)

    def clear_motion(self) -> None:
        self.nodes.motion = self.drop(self.nodes.motion)

    def clear_body(self) -> None:
        self.nodes.body = self.drop(self.nodes.body)

    def clear_all(self) -> None:
        self.clear_motion()
        self.clear_rest()
        self.clear_body()
