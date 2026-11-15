import numpy as np, pandas as pd, json, os
from scipy.signal import butter, sosfiltfilt, find_peaks
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt


D = "../Data/Sensors/walk3_loop/" 
KNOWN_DISTANCE = None            
BIAS_SECS = 5                     
ALIGN_M = 8                       
SCALE = 10                        


rd = lambda n: pd.read_csv(D + n + ".csv")
acc, gyr, ori, loc = rd("Accelerometer"), rd("Gyroscope"), rd("Orientation"), rd("Location")
loc = loc.dropna(subset=["latitude", "longitude"])

t0 = max(d.seconds_elapsed.min() for d in (acc, gyr, ori, loc))
t1 = min(d.seconds_elapsed.max() for d in (acc, gyr, ori, loc))
t = np.arange(t0, t1, 0.02)
I = lambda d, c: np.interp(t, d.seconds_elapsed, d[c])


lat0, lon0 = loc.latitude.iloc[0], loc.longitude.iloc[0]
gx = np.interp(t, loc.seconds_elapsed, (loc.longitude - lon0) * np.cos(np.radians(lat0)) * 111320)
gy = np.interp(t, loc.seconds_elapsed, (loc.latitude - lat0) * 110540)
DIST = KNOWN_DISTANCE or np.hypot(np.diff(gx), np.diff(gy)).sum()


mag = np.sqrt(I(acc, "x")**2 + I(acc, "y")**2 + I(acc, "z")**2)
mag = sosfiltfilt(butter(4, 4, fs=50, output="sos"), mag - mag.mean())
pk, _ = find_peaks(mag, height=0.8 * mag.std(), distance=15)
N = len(pk); L = DIST / N
print(f"steps: {N} | distance {DIST:.0f} m | step length {L:.2f} m")


yaw = I(ori, "yaw")
if np.abs(yaw).max() > 7: yaw = np.radians(yaw)
gz = I(gyr, "z")
if BIAS_SECS: gz = gz - gz[(t - t[0]) < BIAS_SECS].mean()
raw = {"fused": np.unwrap(yaw)[pk], "gyro-only": np.cumsum(gz * 0.02)[pk]}


i = np.argmax(np.hypot(gx - gx[pk[0]], gy - gy[pk[0]]) > ALIGN_M)
b0 = np.arctan2(gx[i] - gx[pk[0]], gy[i] - gy[pk[0]])

def run(h):
    h = h + (b0 - np.angle(np.mean(np.exp(1j * h[:5]))))      
    e = gx[pk[0]] + np.cumsum(L * np.sin(h)); n = gy[pk[0]] + np.cumsum(L * np.cos(h))
    return e, n, np.hypot(e - gx[pk], n - gy[pk]), h

res = {}
for name, h in raw.items():
    best = None
    for sg in (1, -1):
        e, n, err, hh = run(sg * h)
        print(f"{name} sign {sg:+d}: final {err[-1]:.1f} m, mean {err.mean():.1f} m")
        if best is None or err.mean() < best[2].mean(): best = (e, n, err, hh, sg)
    res[name] = best
    print(f"  -> {name}: sign {best[4]:+d} chosen (lowest mean error on this walk)")
    print(f"  {name}: final {best[2][-1]:.1f} m = {100*best[2][-1]/DIST:.1f}% | mean {best[2].mean():.1f} m = {100*best[2].mean()/DIST:.1f}%")


e, n, _, _, _ = res["gyro-only"]
a = np.c_[e - e.mean(), n - n.mean()]; g = np.c_[gx[pk] - gx[pk].mean(), gy[pk] - gy[pk].mean()]
print("best-fit rotation DR->GPS (deg):", np.degrees(np.arctan2((a[:,0]*g[:,1]-a[:,1]*g[:,0]).sum(), (a*g).sum())).round(1), "(near 0 is good)")
print("DR extent:", np.ptp(e).round(1), np.ptp(n).round(1), "| GPS extent:", np.ptp(gx[pk]).round(1), np.ptp(gy[pk]).round(1))
print("loop-closure check: GPS end-to-start", np.hypot(gx[pk[-1]]-gx[pk[0]], gy[pk[-1]]-gy[pk[0]]).round(1), "m")

h2 = res["gyro-only"][3].copy()
for k in (N // 3, 2 * N // 3):
    lo, hi = max(k - 5, 0), min(k + 5, N - 1)
    bg = np.arctan2(gx[pk[hi]] - gx[pk[lo]], gy[pk[hi]] - gy[pk[lo]])
    h2[k:] += np.angle(np.exp(1j * (bg - h2[k])))
e2 = gx[pk[0]] + np.cumsum(L * np.sin(h2)); n2 = gy[pk[0]] + np.cumsum(L * np.cos(h2))
err2 = np.hypot(e2 - gx[pk], n2 - gy[pk])
print(f"gyro-only + 2 heading fixes: final {err2[-1]:.1f} m = {100*err2[-1]/DIST:.1f}% | mean {err2.mean():.1f} m")


fig, ax = plt.subplots(1, 2, figsize=(13, 6))
ax[0].plot(gx[pk[0]:pk[-1]], gy[pk[0]:pk[-1]], "k-", lw=2, label="GPS (reference)")
for (name, r), c in zip(res.items(), ("tab:orange", "tab:blue")):
    ax[0].plot(r[0], r[1], c=c, label=f"{name}: {100*r[2][-1]/DIST:.1f}% final error")
ax[0].plot(e2, n2, "g--", label=f"gyro-only + 2 fixes: {100*err2[-1]/DIST:.1f}%")
ax[0].set_aspect("equal"); ax[0].legend(fontsize=8); ax[0].set_title("Dead reckoning vs GPS"); ax[0].set_xlabel("east (m)"); ax[0].set_ylabel("north (m)")
x = np.arange(1, N + 1) * L
for name, r in res.items(): ax[1].plot(x, r[2], label=name)
ax[1].plot(x, err2, "g--", label="gyro-only + fixes")
ax[1].set_xlabel("distance walked (m)"); ax[1].set_ylabel("position error (m)"); ax[1].legend(); ax[1].set_title("Error vs distance")
plt.savefig("drift_result.png", dpi=140, bbox_inches="tight"); print("saved drift_result.png")


try:
    rt = json.load(open("viewer/route.json")); e0, n0 = rt["waypoints"][0]["e"], rt["waypoints"][0]["n"]
except Exception:
    e0, n0 = 300, 400
off = lambda E, Nn: [[round(e0 + SCALE*(a - gx[pk[0]]), 1), round(n0 + SCALE*(b - gy[pk[0]]), 1)] for a, b in zip(E, Nn)]
json.dump({"scale": SCALE, "start": [e0, n0], "gps": off(gx[pk[0]:pk[-1]:5], gy[pk[0]:pk[-1]:5]),
           "dr_fused": off(res["fused"][0], res["fused"][1]), "dr_gyro": off(res["gyro-only"][0], res["gyro-only"][1])},
          open("../viewer/walk.json", "w"))
print("wrote viewer/walk.json")

print("--- loop closure (true answer = 0 m; no GPS used) ---")
for nm, h in raw.items():
    for sg in (1, -1):
        e_, n_, _, _ = run(sg*h)
        c = np.hypot(e_[-1]-e_[0], n_[-1]-n_[0])
        print(f"{nm} sign {sg:+d}: closure error {c:.1f} m = {100*c/DIST:.1f}% of {DIST:.0f} m")


fig, ax = plt.subplots(figsize=(7, 5))
for nm, h, c in (("fused heading (Earth only)", raw["fused"], "tab:blue"),
                 ("gyro-only heading (Mars-like)", raw["gyro-only"], "tab:orange")):
    h = h - np.angle(np.mean(np.exp(1j*h[:5])))
    e_, n_ = np.cumsum(L*np.sin(h)), np.cumsum(L*np.cos(h))
    cl = np.hypot(e_[-1], n_[-1])
    ax.plot(np.r_[0, e_], np.r_[0, n_], c=c, label=f"{nm}: closure error {cl:.1f} m")
    ax.plot(e_[-1], n_[-1], "x", c=c, ms=10)
ax.plot(0, 0, "ko", label="start = true end")
ax.set_aspect("equal"); ax.legend(fontsize=8); ax.set_xlabel("east (m)"); ax.set_ylabel("north (m)")
ax.set_title("Dead reckoning, closed ~50 m loop")
plt.savefig("drift_closure.png", dpi=140, bbox_inches="tight")