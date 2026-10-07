"""Optional module entry point; use the root run.py as the single configuration."""
from pathlib import Path
import importlib.util
import sys
from .launcher import main as launch


def main(argv=None):
    path = Path.cwd() / "run.py"
    if not path.is_file():
        print("Run from the repository root, or invoke its run.py directly.", file=sys.stderr)
        return 2
    spec = importlib.util.spec_from_file_location("dats_local_settings", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return launch(module.SETTINGS, path.parent, argv)
