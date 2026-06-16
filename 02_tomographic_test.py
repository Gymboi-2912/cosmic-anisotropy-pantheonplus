"""
02_tomographic_test.py
----------------------
Tests whether the H0 hemisphere asymmetry decays with redshift depth,
as expected for a local bulk-flow artefact.

Runs the hemisphere scan independently in three redshift shells:
  Shell 1:  0.023 < z < 0.06   (nearest, N ~ 344)
  Shell 2:  0.06  < z < 0.10   (middle,  N ~ 63)
  Shell 3:  0.10  < z < 0.15   (deep,    N ~ 85)

Both zCMB (uncorrected) and zHD (flow-corrected) frames are tested.
Significance via 500 shuffles per shell (sufficient given small N).

Reproduces Section 3.2 and Figure 3 of Sharma (2026).

Outputs:
    tomo_results.npy  —  dict keyed by (frame, shell_label)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA_FILE = "Pantheon+SH0ES.dat"
COV_FILE  = "Pantheon+SH0ES_STAT+SYS.cov"

C, Q0, J0, H0R = 299792.458, -0.55, 1.0, 70.0

print("Loading data...")
df  = pd.read_csv(DATA_FILE, sep=r'\s+')
raw = np.loadtxt(COV_FILE, skiprows=1)
C_full = raw.reshape(1701, 1701)

k   = np.arange(48)
phi = np.pi * (3 - np.sqrt(5)) * k
ct  = 1 - 2 * (k + 0.5) / 48
st  = np.sqrt(1 - ct**2)
DIRS = np.column_stack([st * np.cos(phi), st * np.sin(phi), ct])

SHELLS = [
    (0.023, 0.06,  '0.023–0.06'),
    (0.06,  0.10,  '0.06–0.10'),
    (0.10,  0.15,  '0.10–0.15'),
]

def mu_cosmo(z):
    dL = (C * z / H0R) * (1 + (1 - Q0)*z/2 - (1 - Q0 - 3*Q0**2 + J0)*z**2/6)
    return 5 * np.log10(dL) + 25

def fit_H0(d, Cinv, sel):
    Cs  = Cinv[np.ix_(sel, sel)]
    one = np.ones(sel.sum())
    x   = -(one @ Cs @ d[sel]) / (one @ Cs @ one)
    return H0R * 10**(x / 5)

def scan_max(nvec, d, Cinv, min_n):
    best = 0.0
    for j in range(48):
        tw = (nvec @ DIRS[j]) > 0
        aw = ~tw
        if tw.sum() < min_n or aw.sum() < min_n:
            continue
        dh = abs(fit_H0(d, Cinv, tw) - fit_H0(d, Cinv, aw))
        if dh > best:
            best = dh
    return best

results = {}
print(f"\n{'Frame':6s} {'Shell':12s} {'N':>4s} {'obs|ΔH0|':>9s} {'p':>6s} {'null95':>7s}")
print("-" * 50)

for frame in ['zCMB', 'zHD']:
    for zlo, zhi, label in SHELLS:
        mask = ((df[frame] > zlo) & (df[frame] < zhi)
                & (df.IS_CALIBRATOR == 0)).values
        idx  = np.where(mask)[0]
        n    = idx.sum()
        if n < 40:
            print(f"{frame:6s} {label:12s} {n:4d}  insufficient data")
            continue

        z   = df[frame].values[mask]
        mu  = df.MU_SH0ES.values[mask]
        ra  = np.radians(df.RA.values[mask])
        dec = np.radians(df.DEC.values[mask])
        Cinv = np.linalg.inv(C_full[np.ix_(idx, idx)])
        nvec = np.column_stack([np.cos(dec)*np.cos(ra),
                                np.cos(dec)*np.sin(ra),
                                np.sin(dec)])
        d = mu - mu_cosmo(z)
        min_n = max(15, int(0.12 * n))

        obs  = scan_max(nvec, d, Cinv, min_n)
        rng  = np.random.default_rng(42)
        null = np.array([scan_max(nvec[rng.permutation(n)], d, Cinv, min_n)
                         for _ in range(500)])
        p    = (np.sum(null >= obs) + 1) / 501
        p95  = np.percentile(null, 95)

        results[(frame, label)] = {'obs': obs, 'null': null, 'p': p,
                                   'p95': p95, 'n': n}
        print(f"{frame:6s} {label:12s} {n:4d}  {obs:9.3f}  {p:6.3f}  {p95:7.3f}")

np.save('tomo_results.npy', results, allow_pickle=True)

# ── Plot ───────────────────────────────────────────────────────────────────────
labels = [s[2] for s in SHELLS]
x = np.arange(len(labels))
fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), dpi=120, sharey=True)
for ax, frame, col in zip(axes, ['zCMB', 'zHD'], ['#D85A30', '#1D9E75']):
    amps = [results.get((frame, s[2]), {}).get('obs', 0) for s in SHELLS]
    n95  = [results.get((frame, s[2]), {}).get('p95', 0) for s in SHELLS]
    ps   = [results.get((frame, s[2]), {}).get('p', 1)  for s in SHELLS]
    ax.bar(x, amps, width=0.5, color=col, alpha=0.85)
    ax.plot(x, n95, 'k_', markersize=30, markeredgewidth=2, label='Noise ceiling (95%)')
    for i, (a, pv) in enumerate(zip(amps, ps)):
        ax.text(i, a + 0.08, f'p={pv:.2f}', ha='center', fontsize=9,
                fontweight='bold' if pv < 0.01 else 'normal')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_title(f'{frame} frame', fontsize=11)
    ax.grid(alpha=0.2, axis='y')
    ax.legend(fontsize=8)
axes[0].set_ylabel('Max |ΔH0| (km/s/Mpc)', fontsize=10)
plt.suptitle('Tomographic depth test', fontsize=11)
plt.tight_layout()
plt.savefig('tomographic_test.png', dpi=150)
print("\nPlot saved: tomographic_test.png")
print("Done.")
