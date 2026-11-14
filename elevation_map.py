import rasterio
import numpy as np
import matplotlib.pyplot as plt
from rasterio.enums import Resampling
import math

path = "Data/Mars/NES_JEZ_MID_DEM_Mosaic_HiRISE_CTX_HRSC_GCS_MARS_07-10-2018(1).tif"

with rasterio.open(path) as ds:

    scale = max(1, math.ceil(max(ds.width, ds.height) / 2000))

    out_height = max(1, ds.height // scale)
    out_width = max(1, ds.width // scale)

    elevation = ds.read(
        1,
        out_shape=(out_height, out_width),
        resampling=Resampling.bilinear,
        masked=True
    )

    bounds = ds.bounds

    valid_values = elevation.compressed()

    vmin, vmax = np.percentile(valid_values, [2, 98])

    plt.figure(figsize=(12, 9))

    image = plt.imshow(
        elevation,
        cmap="terrain",
        extent=(bounds.left, bounds.right,
                bounds.bottom, bounds.top),
        origin="upper",
        vmin=vmin,
        vmax=vmax
    )

    plt.colorbar(image, label="Elevation (raster units)")
    plt.title("Mars Digital Terrain Model - Jezero Crater")
    plt.xlabel("Projected X coordinate")
    plt.ylabel("Projected Y coordinate")

    plt.tight_layout()
    plt.savefig("mars_elevation_map.png", dpi=300)
    plt.show()
