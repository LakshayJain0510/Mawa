import json
import math
import os

import numpy as np
import rasterio
from PIL import Image
from pyproj import CRS as PCRS, Transformer
from rasterio.crs import CRS as RCRS
from rasterio.enums import Resampling
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds

DEM_PATH = "../Data/Mars/JEZ_hirise_soc_006_DTM_MOLAtopography_DeltaGeoid_1m_Eqc_latTs0_lon0_blend40(1).tif"
IMG_PATH = "../Data/Mars/JEZ_hirise_soc_006_orthoMosaic_25cm_Eqc_latTs0_lon0_first_dd.tif"  

WEST, SOUTH, EAST, NORTH = 77.385, 18.416, 77.455, 18.484

GRID = 513                  
TEX = 2048                 
FALLBACK_RADIUS_M = 3396190.0   
OUT_DIR = "viewer"


def pick_resampling(window, target):
    return Resampling.average if max(window.width, window.height) > target else Resampling.bilinear


os.makedirs(OUT_DIR, exist_ok=True)

with rasterio.open(DEM_PATH) as dem:
    print("DEM CRS:", dem.crs)
    print("DEM size:", dem.width, "x", dem.height, "| pixel size:", dem.res, "| dtype:", dem.dtypes[0])
    print("DEM bounds:", dem.bounds)
    if dem.crs is None:
        raise SystemExit("This DEM has no coordinate system, so it can't be placed on the map.")
    if dem.dtypes[0] == "uint8":
        print("WARNING: 8-bit data usually means an IMAGE, not an elevation model. Check DEM_PATH.")

    pcrs = PCRS.from_wkt(dem.crs.to_wkt())
    geo = pcrs.geodetic_crs                        
    to_dem = Transformer.from_crs(geo, pcrs, always_xy=True)
    to_deg = Transformer.from_crs(pcrs, geo, always_xy=True)

    corners = [to_dem.transform(lon, lat)
               for lon, lat in ((WEST, SOUTH), (EAST, SOUTH), (WEST, NORTH), (EAST, NORTH))]
    left, right = min(c[0] for c in corners), max(c[0] for c in corners)
    bottom, top = min(c[1] for c in corners), max(c[1] for c in corners)

    b = dem.bounds
    if left < b.left or right > b.right or bottom < b.bottom or top > b.top:
        lo = to_deg.transform(b.left, b.bottom)
        hi = to_deg.transform(b.right, b.top)
        raise SystemExit(
            "Your study area is outside the DEM. This DEM covers longitude %.4f to %.4f and "
            "latitude %.4f to %.4f (degrees). Edit WEST/SOUTH/EAST/NORTH to fit inside."
            % (lo[0], hi[0], lo[1], hi[1]))

    win = from_bounds(left, bottom, right, top, dem.transform)
    print("Study area covers about %d x %d DEM pixels" % (win.width, win.height))
    z = dem.read(1, window=win, out_shape=(GRID, GRID),
                 resampling=pick_resampling(win, GRID), masked=True)
    z = z.astype("float32").filled(np.nan)
    z[np.abs(z) > 1e6] = np.nan                      

    nan_frac = float(np.isnan(z).mean())
    if nan_frac >= 1.0:
        raise SystemExit("The study area contains no valid elevation data. Move the box.")
    if nan_frac > 0:
        print("Warning: %.1f%% of the study area has no elevation data (gaps in the mosaic)."
              % (100 * nan_frac))
        if nan_frac > 0.03:
            print("  That is a lot. Move the box so it sits fully inside the covered area.")
        z = np.where(np.isnan(z), np.nanmean(z), z)

    radius = geo.ellipsoid.semi_major_metre if geo.ellipsoid else FALLBACK_RADIUS_M
    print("Planet radius used for distances: %.0f m" % radius)


lat_mid = (NORTH + SOUTH) / 2
m_per_deg = math.pi / 180 * radius
width_m = (EAST - WEST) * m_per_deg * math.cos(math.radians(lat_mid))
height_m = (NORTH - SOUTH) * m_per_deg

zmin, zmax = float(z.min()), float(z.max())

dx, dy = width_m / (GRID - 1), height_m / (GRID - 1)
gy, gx = np.gradient(z, dy, dx)
slope = np.degrees(np.arctan(np.hypot(gx, gy)))
print("Study area: %.2f km x %.2f km | grid spacing %.1f m x %.1f m"
      % (width_m / 1000, height_m / 1000, dx, dy))
print("Height values: %.1f to %.1f (relief %.1f m)" % (zmin, zmax, zmax - zmin))
print("  (if the relief is tens to a few hundred metres, that is real terrain;")
print("   the absolute offset doesn't matter, because only differences are used)")
print("Slope: median %.1f deg, 95th percentile %.1f deg, max %.1f deg"
      % (np.median(slope), np.percentile(slope, 95), slope.max()))
for lim in (10, 15, 25):
    print("  cells steeper than %d deg: %.1f%%" % (lim, 100 * (slope > lim).mean()))

z.astype("<f4").tofile(os.path.join(OUT_DIR, "dem.bin"))
meta = {
    "grid": GRID,
    "width_m": width_m,
    "height_m": height_m,
    "zmin": zmin,
    "zmax": zmax,
    "bounds": {"west": WEST, "south": SOUTH, "east": EAST, "north": NORTH},
    "note": "dem.bin = float32 little-endian, row-major, row 0 = north edge",
}
with open(os.path.join(OUT_DIR, "meta.json"), "w") as f:
    json.dump(meta, f, indent=2)
print("Wrote dem.bin and meta.json")

if IMG_PATH:
    with rasterio.open(IMG_PATH) as img:
        print("Image CRS:", img.crs, "| pixel size:", img.res, "| dtype:", img.dtypes[0])
        ib = transform_bounds(RCRS.from_wkt(geo.to_wkt()), img.crs, WEST, SOUTH, EAST, NORTH)
        ibd = img.bounds
        if ib[0] < ibd.left or ib[2] > ibd.right or ib[1] < ibd.bottom or ib[3] > ibd.top:
            print("Warning: the image does not fully cover the study area; part of the texture will be wrong.")
        iwin = from_bounds(*ib, img.transform)
        data = img.read(out_shape=(img.count, TEX, TEX), window=iwin,
                        resampling=pick_resampling(iwin, TEX), masked=True)
    mask = np.ma.getmaskarray(data).any(axis=0)
    bands = data.filled(0).astype("float32")
    rgb = bands[:3] if bands.shape[0] >= 3 else np.repeat(bands[:1], 3, axis=0)
    valid = ~mask
    print("Image coverage of the study area: %.1f%%" % (100 * valid.mean()))
    if valid.any():
        lo, hi = np.percentile(rgb[:, valid], (2, 98))         
        rgb = np.clip((rgb - lo) / max(hi - lo, 1e-6), 0, 1)
        rgb[:, ~valid] = 0
    Image.fromarray((rgb.transpose(1, 2, 0) * 255).astype("uint8")).save(
        os.path.join(OUT_DIR, "texture.jpg"), quality=90)
    print("Wrote texture.jpg (if it looks black, the image doesn't cover the study area)")
