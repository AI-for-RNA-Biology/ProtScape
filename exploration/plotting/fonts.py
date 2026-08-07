"""Register Arial before rendering plots."""

from pathlib import Path

from matplotlib import font_manager


def register_arial() -> None:
    font_dir = Path.home() / ".local/share/fonts/msttcorefonts"
    for path in font_dir.glob("Arial*.TTF"):
        font_manager.fontManager.addfont(str(path))
    for path in font_manager.findSystemFonts():
        if Path(path).stem.lower().startswith("arial"):
            font_manager.fontManager.addfont(str(path))
    try:
        font_manager.findfont("Arial", fallback_to_default=False)
    except ValueError as exc:
        raise RuntimeError("Arial must be installed to render the plots.") from exc
