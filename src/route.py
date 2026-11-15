import json, heapq, math
import numpy as np, rasterio
from rasterio.windows import from_bounds
from rasterio.enums import Resampling
from pyproj import Transformer
from scipy.ndimage import gaussian_filter
from sklearn.cluster import KMeans
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEM_PATH = "../Data/Mars/JEZ_hirise_soc_006_DTM_MOLAtopography_DeltaGeoid_1m_Eqc_latTs0_lon0_blend40(1).tif"
IMG_PATH = "../Data/Mars/JEZ_hirise_soc_006_orthoMosaic_25cm_Eqc_latTs0_lon0_first_dd.tif"  
WEST, SOUTH, EAST, NORTH = 77.385, 18.416, 77.455, 18.484   
META = "../viewer/meta.json"
CELL_M = 4          
MAX_SLOPE = 20.0    
K = 4               
RISKY = [2]         
RISK_W = 6          

WAYPOINTS = [("Start", 300, 400), ("Waypoint A", 3000, 3200)]


meta = json.load(open(META))
W_M, H_M = meta["width_m"], meta["height_m"]


with rasterio.open(DEM_PATH) as src:
    tf = Transformer.from_crs(src.crs.geodetic_crs, src.crs, always_xy=True)
    x0, y0 = tf.transform(WEST, SOUTH)
    x1, y1 = tf.transform(EAST, NORTH)
    win = from_bounds(x0, y0, x1, y1, src.transform)
    z = src.read(1, window=win, masked=True).astype("float32").filled(np.nan)
    res_x, res_y = src.res
nr0, nc0 = z.shape
bad = np.isnan(z)
print("native crop:", z.shape, "| nodata %:", round(100 * bad.mean(), 2))
z[bad] = np.nanmean(z)

lat0 = math.radians((SOUTH + NORTH) / 2)
dx, dy = res_x * math.cos(lat0), res_y
z = gaussian_filter(z, 2)
gy_, gx_ = np.gradient(z, dy, dx)
slope = np.degrees(np.arctan(np.hypot(gx_, gy_)))
slope_clean = slope.copy()                    
slope[bad] = 90                               
print("slope percentiles 50/90/99:", np.percentile(slope, [50, 90, 99]).round(1))


f = max(1, round(CELL_M / dy))
nr, nc = nr0 // f, nc0 // f
blockmax = lambda a: a[:nr*f, :nc*f].reshape(nr, f, nc, f).max(axis=(1, 3))
blockmean = lambda a: a[:nr*f, :nc*f].reshape(nr, f, nc, f).mean(axis=(1, 3))
s = blockmax(slope)                            
slope_mean = blockmean(slope_clean)
rough = gaussian_filter(np.abs(slope_clean - gaussian_filter(slope_clean, 5)), 2)
rough_c = blockmean(rough)
cw, ch = W_M / nc, H_M / nr
print(f"routing grid {nr}x{nc}, cell {cw:.1f} x {ch:.1f} m")


with rasterio.open(IMG_PATH) as isrc:
    itf = Transformer.from_crs(isrc.crs.geodetic_crs, isrc.crs, always_xy=True)
    ix0, iy0 = itf.transform(WEST, SOUTH)
    ix1, iy1 = itf.transform(EAST, NORTH)
    iw = from_bounds(ix0, iy0, ix1, iy1, isrc.transform)
    img = isrc.read(1, window=iw, out_shape=(nr * 4, nc * 4), masked=True,
                    resampling=Resampling.average).astype("float32").filled(np.nan)
if np.isnan(img).all():
    raise SystemExit("HiRISE image does not cover the study box. Use the CTX image as IMG_PATH.")
img = np.where(np.isnan(img), np.nanmean(img), img)
blk = img.reshape(nr, 4, nc, 4)
bright, texture = blk.mean(axis=(1, 3)), blk.std(axis=(1, 3))

X = np.c_[slope_mean.ravel(), rough_c.ravel(), bright.ravel(), texture.ravel()]
X = (X - X.mean(0)) / (X.std(0) + 1e-9)
labels = KMeans(K, n_init=10, random_state=0).fit_predict(X).reshape(nr, nc)
for k in range(K):
    m = labels == k
    print(f"cluster {k}: {100*m.mean():4.0f}% of area | slope {slope_mean[m].mean():5.1f} deg | "
          f"roughness {rough_c[m].mean():.2f} | brightness {bright[m].mean():6.1f} | texture {texture[m].mean():5.1f}")
fig, ax = plt.subplots(1, 2, figsize=(14, 7))
a0 = ax[0].imshow(labels, extent=[0, W_M, 0, H_M], cmap="tab10", vmin=0, vmax=9)
ax[0].set_title("k-means terrain clusters"); plt.colorbar(a0, ax=ax[0], label="cluster id")
ax[1].imshow(bright, extent=[0, W_M, 0, H_M], cmap="gray"); ax[1].set_title("HiRISE brightness (for comparison)")
plt.savefig("clusters.png", dpi=120, bbox_inches="tight"); plt.close()

if not RISKY:   
    RISKY = [int(np.argmax([slope_mean[labels == k].mean() for k in range(K)]))]
print("risky cluster(s):", RISKY)
risk = gaussian_filter(np.isin(labels, RISKY).astype(float), 1)



base = 1 + 8 * (s / MAX_SLOPE) ** 2
cost_base = np.where(s > MAX_SLOPE, np.inf, base)
cost_ml = np.where(s > MAX_SLOPE, np.inf, base + RISK_W * risk)
print("impassable cells %:", round(100 * np.isinf(cost_base).mean(), 1))

def to_cell(e, n):
    return min(nr - 1, max(0, int((H_M - n) / ch))), min(nc - 1, max(0, int(e / cw)))

def to_en(r, c):
    return (c + 0.5) * cw, H_M - (r + 0.5) * ch

def snap(rc):
    if np.isfinite(cost_base[rc]): return rc
    pts = np.argwhere(np.isfinite(cost_base))
    return tuple(pts[np.argmin(((pts - rc) ** 2).sum(1))])


NB = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
def astar(cost, a, b):
    def h(r, c):
        ex, ey = abs(c - b[1]) * cw, abs(r - b[0]) * ch
        return max(ex, ey) + 0.414 * min(ex, ey)       
    g = np.full(cost.shape, np.inf); g[a] = 0
    par = {}; pq = [(h(*a), a)]
    while pq:
        _, (r, c) = heapq.heappop(pq)
        if (r, c) == b: break
        for dr, dc in NB:
            rr, cc = r + dr, c + dc
            if 0 <= rr < nr and 0 <= cc < nc and np.isfinite(cost[rr, cc]):
                step = math.hypot(dc * cw, dr * ch) * (cost[r, c] + cost[rr, cc]) / 2
                if g[r, c] + step < g[rr, cc]:
                    g[rr, cc] = g[r, c] + step; par[(rr, cc)] = (r, c)
                    heapq.heappush(pq, (g[rr, cc] + h(rr, cc), (rr, cc)))
    if b not in par and a != b: return None
    path = [b]
    while path[-1] != a: path.append(par[path[-1]])
    return path[::-1]

def length(pts):
    return sum(math.hypot((q[1]-p[1])*cw, (q[0]-p[0])*ch) for p, q in zip(pts, pts[1:]))

def line_cells(a, b):
    n = max(abs(b[0]-a[0]), abs(b[1]-a[1])) + 1
    return [(round(a[0]+(b[0]-a[0])*t), round(a[1]+(b[1]-a[1])*t)) for t in np.linspace(0, 1, n)]

def plan(cost):
    full = []
    for i in range(len(cells) - 1):
        p = astar(cost, cells[i], cells[i + 1])
        if p is None:
            print(f"NO ROUTE between {WAYPOINTS[i][0]} and {WAYPOINTS[i+1][0]}"); continue
        full += p if not full else p[1:]
    return full

cells = [snap(to_cell(e, n)) for _, e, n in WAYPOINTS]
route_base = plan(cost_base)
route_ml = plan(cost_ml) if (RISKY and RISK_W > 0) else route_base
if not route_base:
    raise SystemExit("No route found. Move the waypoints or raise MAX_SLOPE.")

straight = sum([line_cells(cells[i], cells[i+1]) for i in range(len(cells)-1)], [])
def report(name, p):
    print(f"{name:22s} length {length(p):6.0f} m | max slope {max(s[q] for q in p):5.1f} deg | "
          f"mean risk {np.mean([risk[q] for q in p]):.3f} | max risk {max(risk[q] for q in p):.2f}")
print("\n--- route comparison ---")
print(f"{'straight line':22s} length {length(straight):6.0f} m | max slope {max(s[q] for q in straight):5.1f} deg | "
      f"mean risk {np.mean([risk[q] for q in straight]):.3f}")
report("A* slope-only", route_base)
report("A* slope + ML risk", route_ml)
same = route_base == route_ml
print("ML route differs from slope-only route:", not same)
if same and RISKY:
    print(">> Route did not change. Raise RISK_W (e.g. 15) or pick a cluster that lies between start and goal.")


def export(path_cells, fname):
    pts = [to_en(*c) for c in path_cells]
    pts = pts[::2] + ([pts[-1]] if len(pts) % 2 == 0 else [])
    json.dump({"points": [[round(e, 1), round(n, 1)] for e, n in pts],
               "waypoints": [{"name": nm, "e": e, "n": n} for nm, e, n in WAYPOINTS]},
              open(fname, "w"), indent=1)
export(route_ml, "../viewer/route.json")           
export(route_base, "../viewer/route_baseline.json")
print("wrote viewer/route.json (", "ML" if not same else "baseline", ") and viewer/route_baseline.json")


fig, ax = plt.subplots(figsize=(10, 10))
im = ax.imshow(s, extent=[0, W_M, 0, H_M], cmap="magma", vmax=40)
plt.colorbar(im, label="slope (deg)")
if RISKY:
    ax.contour(np.linspace(0, W_M, nc), np.linspace(H_M, 0, nr), risk, levels=[0.5], colors="yellow", linewidths=0.8)
ax.plot(*zip(*[to_en(*q) for q in straight]), "w--", lw=1, label="straight line")
ax.plot(*zip(*[to_en(*q) for q in route_base]), "c-", lw=2, label="A* slope-only")
if not same: ax.plot(*zip(*[to_en(*q) for q in route_ml]), "-", c="lime", lw=2, label="A* slope + ML risk")
for nm, e, n in WAYPOINTS: ax.plot(e, n, "o", c="red"); ax.text(e, n, nm, color="white")
ax.set_xlabel("east (m)"); ax.set_ylabel("north (m)"); ax.legend()
plt.savefig("routing_debug.png", dpi=130, bbox_inches="tight")
print("saved routing_debug.png and clusters.png")