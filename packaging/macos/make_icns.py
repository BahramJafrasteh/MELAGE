"""Convert assets/resource/main.ico → assets/resource/main.icns (macOS only)."""
import os
import subprocess
import sys
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    sys.exit("Pillow is required: pip install Pillow")

SRC = Path("assets/resource/main.ico")
ICONSET = Path("MELAGE.iconset")
OUT = Path("assets/resource/main.icns")

SIZES = [16, 32, 128, 256, 512]

ICONSET.mkdir(exist_ok=True)
img = Image.open(SRC).convert("RGBA")

for s in SIZES:
    img.resize((s, s), Image.LANCZOS).save(ICONSET / f"icon_{s}x{s}.png")
    img.resize((s * 2, s * 2), Image.LANCZOS).save(ICONSET / f"icon_{s}x{s}@2x.png")

subprocess.run(["iconutil", "-c", "icns", str(ICONSET), "-o", str(OUT)], check=True)
print(f"Created {OUT}")
