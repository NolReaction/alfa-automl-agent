"""ML agent components: research, model access and experiment execution."""

from pathlib import Path


def project_root() -> Path:
    """Locate helpers in both the source package and the flat contest archive."""
    source = Path(__file__).resolve()
    return source.parent.parent if source.name == "__init__.py" else source.parent
