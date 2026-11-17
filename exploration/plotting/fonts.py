"""Use Arial when installed, with an open-source sans-serif fallback."""

from pathlib import Path

from matplotlib import font_manager


font_dir = Path.home() / ".local/share/fonts/msttcorefonts"
for path in font_dir.glob("Arial*.TTF"):
    font_manager.fontManager.addfont(str(path))

available_fonts = {font.name for font in font_manager.fontManager.ttflist}
FONT_NAME = next(
    name for name in ("Arial", "Liberation Sans", "DejaVu Sans")
    if name in available_fonts
)
