"""Load the BrainCo ABI3 extension without exposing a foreign site-packages.

PC2 currently stores ``bc_stark_sdk`` under a Python 3.10 user site while the
teleop runtime uses Python 3.11.  The extension is ABI3-compatible, but adding
the entire 3.10 site-packages directory to ``PYTHONPATH`` also imports
incompatible pure-Python packages such as ``typing``.  This loader exposes only
the vendor package directory as a namespace package.
"""

from __future__ import annotations

import importlib
import os
import sys
import types
from pathlib import Path
from typing import Any

DEFAULT_BRAINCO_PACKAGE_DIR = Path("/home/unitree/.local/lib/python3.10/site-packages/bc_stark_sdk")


def load_brainco_sdk(package_dir: str | Path | None = None) -> Any:
    """Return ``bc_stark_sdk.main_mod`` using an isolated package path."""

    loaded = sys.modules.get("bc_stark_sdk.main_mod")
    if loaded is not None:
        return loaded

    try:
        return importlib.import_module("bc_stark_sdk.main_mod")
    except ModuleNotFoundError as exc:
        if exc.name not in {"bc_stark_sdk", "bc_stark_sdk.main_mod"}:
            raise

    configured = package_dir or os.environ.get(
        "BC_STARK_SDK_PACKAGE_DIR", DEFAULT_BRAINCO_PACKAGE_DIR
    )
    resolved = Path(configured).expanduser().resolve(strict=True)
    extension_files = tuple(resolved.glob("main_mod*.so"))
    if not extension_files:
        raise ImportError(f"BrainCo main_mod extension not found in {resolved}")

    package = types.ModuleType("bc_stark_sdk")
    package.__file__ = None
    package.__package__ = "bc_stark_sdk"
    package.__path__ = [str(resolved)]
    sys.modules["bc_stark_sdk"] = package
    try:
        return importlib.import_module("bc_stark_sdk.main_mod")
    except Exception:
        sys.modules.pop("bc_stark_sdk", None)
        raise
