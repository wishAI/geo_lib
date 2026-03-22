from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def save_structure_preview(image_rgb: np.ndarray, output_path: str | Path) -> None:
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    if image_rgb.dtype != np.uint8:
        image_rgb = np.clip(image_rgb, 0, 255).astype(np.uint8)

    image = Image.fromarray(image_rgb)
    image.save(target)
