import omni.ui as ui

class UIWidget:
    def __init__(self):
        self._current_target_text = "No current target"
        self._window = None
        self.build_ui()

    def build_ui(self):
        self._window = ui.Window("Camera Target", width=200, height=100)
        with self._window.frame:
                self._label_widget = ui.Label(self._current_target_text)

    def set_overlay_text(self, target:str|None):
        if self._window:
            if target:
                self._current_target_text = target
            else:
                self._current_target_text = "No current target"
            self._label_widget.text = self._current_target_text