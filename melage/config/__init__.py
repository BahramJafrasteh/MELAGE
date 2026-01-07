import json
import os
from pathlib import Path
from sys import platform

ADULT_WEIGHTS_FILE = 'New_model_latest_SegMulti_current.pth'
INFANT_WEIGHTS_FILE = 'New_model_latest_SegMulti_current_neo.pth'
__VERSION__ = '2.1.1'
VERSION_DATE = 'Dec 30 2025'
class SettingsManager:
    """
    A singleton class to manage all application settings.
    It loads defaults, overrides them with user-saved settings,
    and provides a central point for saving changes.
    """
    _instance = None

    @staticmethod
    def instance():
        """Returns the single instance of the SettingsManager."""
        if SettingsManager._instance is None:
            SettingsManager._instance = SettingsManager()
        return SettingsManager._instance

    def __init__(self):
        if SettingsManager._instance is not None:
            raise Exception("This class is a singleton! Use instance().")

        # --- 1. Define Application-Relative Paths (Should NOT change) ---
        # We go up two levels: config.py -> melage -> Project Root
        self.APP_ROOT = Path(__file__).resolve().parent.parent.parent
        self.ASSETS_DIR = self.APP_ROOT / "assets"
        self.RESOURCE_DIR = self.ASSETS_DIR / "resource"
        self.DOCS_DIR = self.APP_ROOT / "docs"
        self.DEFAULT_MODELS_DIR = self.APP_ROOT / "models" / "NetworkWeights"
        self.DATA_DIR = self.APP_ROOT / "data"  # e.g., for MNI templates
        self.PLUGIN_DIR = self.APP_ROOT / "melage"/"plugins"

        self.SCHEME_DIR = str(self.RESOURCE_DIR / "color")
        self.APP_ROOT = str(self.APP_ROOT)
        self.ASSETS_DIR = str(self.ASSETS_DIR)
        self.RESOURCE_DIR = str(self.RESOURCE_DIR)
        self.DOCS_DIR = str(self.DOCS_DIR)
        self.DEFAULT_MODELS_DIR = str(self.DEFAULT_MODELS_DIR)
        self.DATA_DIR = str(self.DATA_DIR)

        # --- 2. Define User-Configurable Settings (with defaults) ---
        self.auto_save_interval = 10.0
        #self.models_dir = self.DEFAULT_MODELS_DIR

        if platform == "linux" or platform == "linux2" or platform == "darwin":
            default_user_dir = Path.home() / "Desktop"
        else:
            default_user_dir = Path(os.environ['USERPROFILE']) / "Desktop"

        self.DEFAULT_USE_DIR = str(default_user_dir)

        # --- 3. Define Settings File Path ---
        # This stores user preferences
        self.settings_file = Path.home() / ".melage_settings.json"

        # --- 4. Load saved settings ---
        self.load()

    def load(self):
        """Loads settings from the JSON file, if it exists."""
        if not self.settings_file.exists():
            return  # No settings saved yet, use defaults

        try:
            with open(self.settings_file, 'r') as f:
                saved_settings = json.load(f)

            # Update instance variables with saved values
            self.auto_save_interval = saved_settings.get('auto_save_interval', self.auto_save_interval)
            self.DEFAULT_MODELS_DIR = str(Path(saved_settings.get('DEFAULT_MODELS_DIR', self.DEFAULT_MODELS_DIR)))
            self.DEFAULT_USE_DIR = str(Path(saved_settings.get('DEFAULT_USE_DIR', self.DEFAULT_USE_DIR)))

        except Exception as e:
            print(f"Error loading settings: {e}. Using defaults.")

    def save(self):
        """Saves the current settings to the JSON file."""
        settings_to_save = {
            'auto_save_interval': self.auto_save_interval,
            'DEFAULT_MODELS_DIR': str(self.DEFAULT_MODELS_DIR),
            'DEFAULT_USE_DIR': str(self.DEFAULT_USE_DIR),
        }

        try:
            with open(self.settings_file, 'w') as f:
                json.dump(settings_to_save, f, indent=4)
        except Exception as e:
            print(f"Error saving settings: {e}")


# --- Global Instance ---
# Any file in your project can now just `from melage.config import settings`
# to get the one and only settings object.
settings = SettingsManager.instance()
