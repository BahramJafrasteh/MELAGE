import os
import importlib.util
from melage.config import settings
import sys
import importlib
import os


def list_available_tools():
    """
    Scans the 'melage.plugins' package for valid tools.
    Returns a dictionary of {tool_id: friendly_name}.
    """
    available_tools = {}

    # 1. Define the base package path
    # This assumes you are running from the root as 'python -m melage.main'
    plugins_package = "melage.plugins"

    # 2. Locate the plugins directory based on the melage package location
    # We find where 'melage' is installed/located to find the 'plugins' folder
    try:
        import melage.plugins
        plugins_path = settings.PLUGIN_DIR
    except ImportError:
        # Fallback if running locally and path isn't perfectly set
        plugins_path = settings.PLUGIN_DIR

    if not plugins_path.exists():
        print(f"Warning: Plugins directory not found at {plugins_path}")
        return {}

    # 3. Iterate over directories in plugins folder
    for item in plugins_path.iterdir():
        if item.is_dir() and not item.name.startswith('__'):
            tool_id = item.name

            # 4. Construct the expected schema module name
            # Pattern: melage.plugins.<tool_folder>.<tool>_schema
            # Example: melage.plugins.change_coord.change_coord_schema
            schema_module_name = f"{plugins_package}.{tool_id}.{tool_id}_schema"

            try:
                # 5. Import as a proper package module
                # This fixes the "relative import" error because Python now knows the parent package.
                mod = importlib.import_module(schema_module_name)

                # Try to grab a friendly name if it exists, otherwise use folder name
                # You might need to check 'schema' dict inside the module depending on code
                if hasattr(mod, 'schema') and isinstance(mod.schema, dict):
                    friendly_name = mod.schema.get('name', tool_id)
                elif hasattr(mod, 'name'):
                    friendly_name = mod.name
                else:
                    friendly_name = tool_id

                available_tools[tool_id] = friendly_name

            except ImportError as e:
                # If a specific plugin is broken, skip it but don't crash the app
                # print(f"Debug: Could not load {tool_id}: {e}")
                pass
            except Exception as e:
                pass

    return available_tools


def get_plugin_runner(tool_name):
    """
    Dynamically imports the correct module and returns the function.
    """
    # Map tool names to their specific implementation locations
    # You might eventually make this fully dynamic, but a map is safe for now.
    runners = {
        'mga_net': 'melage.plugins.mga_net.MGA_Net.run_headless',
        'bet': 'melage.plugins.bet.main.BET.run_headless_bet',
        # Add others here
    }

    if tool_name not in runners:
        return None

    module_path, func_name = runners[tool_name].rsplit('.', 1)

    try:
        mod = importlib.import_module(module_path)
        return getattr(mod, func_name)
    except (ImportError, AttributeError) as e:
        print(f"Debug: Failed to import runner for {tool_name}: {e}")
        return None