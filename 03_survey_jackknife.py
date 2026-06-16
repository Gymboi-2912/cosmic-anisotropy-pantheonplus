"""
03_survey_jackknife.py
----------------------
Removes each of the 16 contributing Pantheon+ surveys in turn and
checks whether the H0 hemisphere asymmetry signal is driven by any
single survey's calibration or sky concentration.

Uses the zCMB frame (where the signal is significant).

Reproduces Section 3.3 of Sharma (2026).

Outputs:
    survey_jackknife.npy  —  dict with per-survey results
"""

import numpy as np
import pandas as pd

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

def mu_cosmo(z):
    dL = (C * z / H0R) * (1 + (1 - Q0)*z/2 - (1 - Q0 - 3*Q0**2 + J0)*z**2/6)
    return 5 * np.log10(dL) + 25

def fit_H0(d, Cinv, sel):
    Cs  = Cinv[np.ix_(sel, sel)]
    one = np.ones(sel.sum())
    x   = -(one @ Cs @ d[sel]) / (one @ Cs @ one)
    return H0R * 10**(x / 5)

def scan_max(nvec, d, Cinv, min_n=50):
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

# ── Full sample baseline ────────────────────────────────────────────────────────
base_mask = ((df.zCMB > 0.023) & (df.zCMB < 0.15)
             & (df.IS_CALIBRATOR == 0)).values
base_idx  = np.where(base_mask)[0]
z_b  = df.zCMB.values[base_mask]
mu_b = df.MU_SH0ES.values[base_mask]
ra_b = np.radians(df.RA.values[base_mask])
dec_b= np.radians(df.DEC.values[base_mask])
Cinv_b = np.linalg.inv(C_full[np.ix_(base_idx, base_idx)])
nvec_b = np.column_stack([np.cos(dec_b)*np.cos(ra_b),
                          np.cos(dec_b)*np.sin(ra_b),
                          np.sin(dec_b)])
d_b    = mu_b - mu_cosmo(z_b)
obs_full = scan_max(nvec_b, d_b, Cinv_b)
print(f"Full sample (N={base_mask.sum()}): max|ΔH0| = {obs_full:.3f} km/s/Mpc\n")

# ── Survey jackknife ────────────────────────────────────────────────────────────
surveys = df.IDSURVEY[base_mask].value_counts()
print(f"{'Survey':>10s}  {'N_removed':>10s}  {'max|ΔH0|':>10s}  {'Change':>8s}  Note")
print("-" * 60)

jack_results = {'full_obs': obs_full}
for sid in surveys.index:
    jack_mask = base_mask & (df.IDSURVEY != sid).values
    jack_idx  = np.where(jack_mask)[0]
    n_rem     = surveys[sid]

    if jack_idx.sum() < 80:
        print(f"{sid:10d}  {n_rem:10d}  insufficient data after removal")
        continue

    z_j   = df.zCMB.values[jack_mask]
    mu_j  = df.MU_SH0ES.values[jack_mask]
    ra_j  = np.radians(df.RA.values[jack_mask])
    dec_j = np.radians(df.DEC.values[jack_mask])
    Cinv_j = np.linalg.inv(C_full[np.ix_(jack_idx, jack_idx)])
    nvec_j = np.column_stack([np.cos(dec_j)*np.cos(ra_j),
                              np.cos(dec_j)*np.sin(ra_j),
                              np.sin(dec_j)])
    d_j   = mu_j - mu_cosmo(z_j)
    obs_j = scan_max(nvec_j, d_j, Cinv_j)
    change = obs_j - obs_full
    flag   = ' ← large change' if abs(change) > 0.3 else ''

    jack_results[sid] = {'n_removed': n_rem, 'obs': obs_j, 'change': change}
    print(f"{sid:10d}  {n_rem:10d}  {obs_j:10.3f}  {change:+8.3f}{flag}")

np.save('survey_jackknife.npy', jack_results, allow_pickle=True)
print("\nDone. Results saved to survey_jackknife.npy")
