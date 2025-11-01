"""Top-level package for lip_token project."""

from importlib import resources as _resources


def get_project_path() -> str:
    """Return the absolute path to the installed project package."""
    with _resources.as_file(_resources.files(__name__)) as package_path:
        return str(package_path)

