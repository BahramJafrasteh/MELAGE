import importlib
import inspect
from pathlib import Path
from melage.widgets import MelagePlugin
import os

class PluginManager:
    """
    Finds, loads, and manages all plugins for the application.
    """

    def __init__(self, plugin_folder: Path):
        self.plugin_folder = plugin_folder
        self.plugins = []  # List of loaded plugin instances
        self.plugin_modules = {}  # To keep modules from being garbage-collected

    def discover_plugins(self):
        """
        Finds and loads all valid plugins from the plugin folder.
        """
        self.plugins = []

        # Ensure the plugin folder exists
        if not self.plugin_folder.exists():
            print(f"Plugin folder not found: {self.plugin_folder}")
            return

        # Add folder to system path to allow imports
        import sys
        sys.path.insert(0, str(self.plugin_folder.parent))
        plugin_dirs = [
            d for d in Path(self.plugin_folder).iterdir()
            if d.is_dir() and not d.name.startswith('_')
        ]
        for dir_plugin in plugin_dirs:
            for file in dir_plugin.glob("*.py"):
                if file.name.startswith("_"):
                    continue  # Skip __init__.py and other private files
                if '_schema' in file.name:
                    continue  # Skip schema files
                module_name = f"{self.plugin_folder.name}.{dir_plugin.name}.{file.stem}"

                try:
                    # Import the file as a Python module
                    module = importlib.import_module(module_name)
                    self.plugin_modules[module_name] = module

                    # Scan the module for classes that implement our "contract"
                    for name, obj in inspect.getmembers(module, inspect.isclass):
                        if issubclass(obj, MelagePlugin) and obj is not MelagePlugin:
                            # Found a plugin! Create an instance.
                            plugin_instance = obj()
                            self.plugins.append(plugin_instance)
                            print(f"Loaded plugin: {plugin_instance.name}")

                except Exception as e:
                    print(f"Error loading plugin from {file.name}: {e}")

    def get_plugins(self) -> list[MelagePlugin]:
        """Returns the list of all loaded plugin instances."""
        return self.plugins

