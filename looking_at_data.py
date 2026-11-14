import glob
import rasterio
from rasterio.enums import Resampling

for path in sorted(glob.glob("Data/Mars/*.tif*")):
    with rasterio.open(path) as ds:
        print("\n", path)
        print(" size (px):", ds.width, "x", ds.height, "| bands:", ds.count, "| dtype:", ds.dtypes[0])
        print(" CRS:", ds.crs)
        print(" pixel size:", ds.res)
        print(" bounds:", ds.bounds)
        print(" nodata:", ds.nodata)
        # read a shrunken copy so a multi-GB file doesn't fill your memory
        scale = max(1, max(ds.width, ds.height) // 1000)
        a = ds.read(1, out_shape=(ds.height // scale, ds.width // scale),
                    resampling=Resampling.nearest).astype("float64")
        if ds.nodata is not None:
            a = a[a != ds.nodata]
        print(" value range (sampled):", a.min(), "to", a.max())