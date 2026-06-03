from PyQt5.QtWidgets import QWidget
from .ui_schema import *
from .ui_builder import UIBuilder

class MelagePlugin:
    """
    The base class "contract" for all MELAGE plugins.

    A plugin must provide its name and a method to create its main widget.
    """

    def __init__(self):
        pass

    @property
    def name(self) -> str:
        """The name of the plugin to be shown in the menu."""
        raise NotImplementedError("Plugin must have a name.")

    @property
    def description(self) -> str:
        """A short description for tooltips."""
        return ""  # Optional

    @property
    def reference(self) -> str:
        """
        Returns a citation, DOI, or URL for the algorithm.
        Supports HTML (e.g., <a href='...'>Link</a>).
        """
        return ""

    @property
    def category(self) -> str:
        """Category to group plugins in (e.g., "Segmentation", "Registration")."""
        return "General"  # Default category

    def get_widget(self, parent=None) -> QWidget:
        """
        The main function. This method must create and return
        the plugin's main QWidget (or QDialog).
        """
        raise NotImplementedError("Plugin must provide a widget.")


from .DockWidgets import dockWidgets
from .EnhanceImageWidget import enhanceIm
from .SettingsWidget import SettingsDialog
from .openglWidgets import openglWidgets
from .plugin_manager import PluginManager
