"""Create animated GIF showing progression across tomographic bins."""

from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

snakemake = globals().get("snakemake")  # type: ignore
if snakemake is None:
    raise RuntimeError("This script must be executed via Snakemake")

method = snakemake.wildcards.method
field = snakemake.wildcards.field

log_fp = None
if snakemake.log:
    log_path = Path(snakemake.log[0])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_fp = log_path.open("w")


def log(message: str) -> None:
    print(message)
    if log_fp is not None:
        print(message, file=log_fp)


# Load all input images for bins 1-6 + all
images = []
for input_path in snakemake.input:
    img = Image.open(input_path)
    images.append(img)

log(f"Loaded {len(images)} frames for {field} map (method={method})")
log(f"Frame sequence: bins 1-6, then all")

# Save as animated GIF
output_path = Path(snakemake.output[0])
output_path.parent.mkdir(parents=True, exist_ok=True)

# Duration in milliseconds per frame (1000ms = 1 second)
duration = 800

images[0].save(
    output_path,
    save_all=True,
    append_images=images[1:],
    duration=duration,
    loop=0,  # 0 = infinite loop
)

log(f"Saved animated GIF to {output_path}")
log(f"Animation: {len(images)} frames at {duration}ms per frame")

if log_fp is not None:
    log_fp.close()
