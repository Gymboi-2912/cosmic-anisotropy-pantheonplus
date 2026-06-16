"""
01_h0_hemisphere_scan.py
------------------------
Measures the hemispherical H0 asymmetry in Pantheon+ supernovae
in both the uncorrected (zCMB) and flow-corrected (zHD) redshift frames.

Uses a Fibonacci lattice of 48 test directions. For each direction,
splits the sample into two hemispheres and fits H0 analytically using
the full covariance matrix. Significance is assessed by 10,000 coordinate
shuffles with the plus-one p-value estimator (Phipson & Smyth 2010).

Reproduces Section 3.1 and Figure 2 of Sharma (2026).

Outputs:
    h0_scan_zCMB.npy  —  {'obs', 'null', 'p', 'p95'}
    h0_scan_zHD.npy   —  {'obs', 'null', 'p', 'p95'}
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── File paths ─────────────────────────────────────────────────────────────────
DATA_FILE = "Pantheon+SH0ES.dat"
COV_FILE  = "Pantheon+SH0ES_STAT+SYS.cov"

# ── Cosmographic parameters (fixed) ────────────────────────────────────────────
C   = 299792.458   # speed of light, km/s
Q0  = -0.55        # deceleration parameter
J0  =  1.0         # jerk parameter
H0R =  70.0        # reference H0 for residual computation

# ── Load data ──────────────────────────────────────────────────────────────────
print("Loading Pantheon+ data...")
df  = pd.read_csv(DATA_FILE, sep=r'\s+')
raw = np.loadtxt(COV_FILE, skiprows=1)
N_FULL = 1701
C_full = raw.reshape(N_FULL, N_FULL)

# ── Fibonacci sky grid (48 directions) ─────────────────────────────────────────
def fibonacci_dirs(n=48):
    k   = np.arange(n)
    phi = np.pi * (3 - np.sqrt(5)) * k
    ct  = 1 - 2 * (k + 0.5) / n
    st  = np.sqrt(1 - ct**2)
    return np.column_stack([st * np.cos(phi), st * np.sin(phi), ct])

DIRS = fibonacci_dirs(48)

def analytic_H0(d_resid, Cinv_sub):
    """Analytic H0 fit using full covariance on a hemisphere subset."""
    one = np.ones(len(d_resid))
    x   = -(one @ Cinv_sub @ d_resid) / (one @ Cinv_sub @ one)
    return H0R * 10**(x / 5)

def hemisphere_scan(nvec, d_resid, Cinv, min_n=50):
    """Scan 48 directions, return max |ΔH0| and the direction index."""
    dots = nvec @ DIRS.T
    best_dh, best_j = 0.0, -1
    for j in range(len(DIRS)):
        tw = dots[:, j] > 0
        aw = ~tw
        if tw.sum() < min_n or aw.sum() < min_n:
            continue
        h_t = analytic_H0(d_resid[tw],  Cinv[np.ix_(tw, tw)])
        h_a = analytic_H0(d_resid[aw],  Cinv[np.ix_(aw, aw)])
        dh  = abs(h_t - h_a)
        if dh > best_dh:
            best_dh, best_j = dh, j
    return best_dh, best_j

def mu_cosmo(z):
    """Cosmographic distance modulus at reference H0."""
    dL = (C * z / H0R) * (1 + (1 - Q0) * z / 2
          - (1 - Q0 - 3 * Q0**2 + J0) * z**2 / 6)
    return 5 * np.log10(dL) + 25

# ── Run scan for each redshift frame ───────────────────────────────────────────
results = {}
for frame in ['zCMB', 'zHD']:
    print(f"\nProcessing {frame} frame...")
    mask = ((df[frame] > 0.023) & (df[frame] < 0.15)
            & (df.IS_CALIBRATOR == 0)).values
    idx  = np.where(mask)[0]

    z   = df[frame].values[mask]
    mu  = df.MU_SH0ES.values[mask]
    ra  = np.radians(df.RA.values[mask])
    dec = np.radians(df.DEC.values[mask])

    Cinv = np.linalg.inv(C_full[np.ix_(idx, idx)])
    nvec = np.column_stack([np.cos(dec) * np.cos(ra),
                            np.cos(dec) * np.sin(ra),
                            np.sin(dec)])
    d_resid = mu - mu_cosmo(z)

    obs, jmax = hemisphere_scan(nvec, d_resid, Cinv)
    print(f"  Observed max |ΔH0| = {obs:.3f} km/s/Mpc")
    print(f"  Best direction: RA={np.degrees(np.arctan2(DIRS[jmax,1],DIRS[jmax,0]))%360:.0f}°"
          f"  DEC={np.degrees(np.arcsin(DIRS[jmax,2])):+.0f}°")

    # 10,000 shuffle test
    print(f"  Running 10,000 shuffles...")
    rng  = np.random.default_rng(42)
    null = np.zeros(10000)
    for t in range(10000):
        perm    = rng.permutation(len(z))
        null[t], _ = hemisphere_scan(nvec[perm], d_resid, Cinv)

    p    = (np.sum(null >= obs) + 1) / 10001   # plus-one estimator
    p95  = np.percentile(null, 95)
    print(f"  p = {p:.4f}  (null 95th pct = {p95:.3f})")

    results[frame] = {'obs': obs, 'null': null, 'p': p, 'p95': p95,
                      'best_dir': DIRS[jmax]}
    np.save(f'h0_scan_{frame}.npy', results[frame], allow_pickle=True)

# ── Quick summary plot ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=120, sharey=False)
for ax, frame in zip(axes, ['zCMB', 'zHD']):
    r = results[frame]
    ax.hist(r['null'], bins=40, color='#8F8F94', alpha=0.85, label='10,000 shuffled skies')
    ax.axvline(r['obs'], c='#D85A30', lw=2.5, label=f"Real sky: {r['obs']:.2f}")
    ax.axvline(r['p95'], c='k', lw=1.5, ls='--', label=f"95th pct: {r['p95']:.2f}")
    ax.set_title(f"{frame}: p = {r['p']:.4f}", fontsize=11)
    ax.set_xlabel('Max |ΔH0| (km/s/Mpc)', fontsize=10)
    ax.set_yticks([])
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
plt.tight_layout()
plt.savefig('h0_hemisphere_scan.png', dpi=150)
print("\nPlot saved: h0_hemisphere_scan.png")
print("Done.")
