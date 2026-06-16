"""
04_profile_likelihood_Om.py
---------------------------
Computes the nonlinear profile likelihood for Ωm in each hemisphere
at the maximum-signal direction from the H0 scan.

For each value of Ωm on a grid, H0 is solved analytically (exact
covariance-weighted solution), and the resulting chi-squared is recorded.
This reveals whether the H0-Ωm parameter space is degenerate at z < 0.15.

Reproduces Section 3.4 and Figure 4 of Sharma (2026).

Key finding: both hemispheres prefer unphysically low Ωm ~ 0.10-0.18,
confirming severe degeneracy. The apparent ΔΩm ~ 0.6 from linearised
fitting is an artefact, not a physical detection.

Outputs:
    profile_likelihood.npy  —  chi2 arrays for both hemispheres
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA_FILE = "Pantheon+SH0ES.dat"
COV_FILE  = "Pantheon+SH0ES_STAT+SYS.cov"

C, H0R = 299792.458, 70.0

# Maximum-signal direction from H0 scan (RA=145°, DEC=-61°)
# This is the direction that produced the largest apparent ΔΩm
DIPOLE_RA  = 145.0
DIPOLE_DEC = -61.0

print("Loading data...")
df  = pd.read_csv(DATA_FILE, sep=r'\s+')
raw = np.loadtxt(COV_FILE, skiprows=1)
C_full = raw.reshape(1701, 1701)

mask = ((df.zHD > 0.023) & (df.zHD < 0.15)
        & (df.IS_CALIBRATOR == 0)).values
idx  = np.where(mask)[0]
z    = df.zHD.values[mask]
mu   = df.MU_SH0ES.values[mask]
ra   = np.radians(df.RA.values[mask])
dec  = np.radians(df.DEC.values[mask])
Cinv = np.linalg.inv(C_full[np.ix_(idx, idx)])
nvec = np.column_stack([np.cos(dec)*np.cos(ra),
                        np.cos(dec)*np.sin(ra),
                        np.sin(dec)])

# Split hemispheres at the dipole direction
d_ax = np.array([np.cos(np.radians(DIPOLE_DEC)) * np.cos(np.radians(DIPOLE_RA)),
                 np.cos(np.radians(DIPOLE_DEC)) * np.sin(np.radians(DIPOLE_RA)),
                 np.sin(np.radians(DIPOLE_DEC))])
tw = (nvec @ d_ax) > 0
aw = ~tw
print(f"Hemisphere sizes: toward={tw.sum()}  away={aw.sum()}")

def mu_lcdm(z, H0, Om, N=60):
    """Full numerical luminosity distance for flat ΛCDM."""
    a_em = 1 / (1 + z)
    dc   = np.zeros(len(z))
    OL   = 1 - Om
    for i in range(len(z)):
        aa = np.linspace(a_em[i], 1.0, N)
        da = np.diff(aa)
        am = 0.5 * (aa[:-1] + aa[1:])
        H  = H0 * np.sqrt(OL + Om * am**(-3)) / am**2
        dc[i] = np.sum(C * da / (am**2 * H))
    return 5 * np.log10((1 + z) * dc) + 25

def profile_chi2(sel, Om):
    """Analytic H0 at fixed Om, return chi2 and best-fit H0."""
    Cs  = Cinv[np.ix_(sel, sel)]
    one = np.ones(sel.sum())
    mu_ref = mu_lcdm(z[sel], H0R, Om)
    d_     = mu[sel] - mu_ref
    x      = (one @ Cs @ d_) / (one @ Cs @ one)
    H0_    = H0R * 10**(x / 5)
    r      = mu[sel] - mu_lcdm(z[sel], H0_, Om)
    return float(r @ Cs @ r), H0_

# ── Profile likelihood grid ────────────────────────────────────────────────────
Om_grid = np.linspace(0.10, 0.60, 50)
chi_t   = np.zeros(50)
chi_a   = np.zeros(50)
H0_t    = np.zeros(50)
H0_a    = np.zeros(50)

print("Computing profile likelihood...")
for i, Om in enumerate(Om_grid):
    chi_t[i], H0_t[i] = profile_chi2(tw, Om)
    chi_a[i], H0_a[i] = profile_chi2(aw, Om)

best_t   = Om_grid[chi_t.argmin()]
best_a   = Om_grid[chi_a.argmin()]
range_t  = chi_t.max() - chi_t.min()
range_a  = chi_a.max() - chi_a.min()

print(f"\nToward hemisphere:")
print(f"  Best-fit Ωm = {best_t:.3f}  (H0 = {H0_t[chi_t.argmin()]:.2f})")
print(f"  Δχ² range across grid = {range_t:.1f}")
print(f"\nAway hemisphere:")
print(f"  Best-fit Ωm = {best_a:.3f}  (H0 = {H0_a[chi_a.argmin()]:.2f})")
print(f"  Δχ² range across grid = {range_a:.1f}")
print(f"\nΔΩm (nonlinear profile) = {best_t - best_a:+.3f}")
print("Note: both hemispheres prefer Ωm near the grid lower bound (0.10).")
print("This indicates severe H0-Ωm degeneracy at z < 0.15.")

np.save('profile_likelihood.npy',
        {'Om_grid': Om_grid, 'chi_t': chi_t, 'chi_a': chi_a,
         'H0_t': H0_t, 'H0_a': H0_a,
         'best_Om_t': best_t, 'best_Om_a': best_a},
        allow_pickle=True)

# ── Plot ───────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4.5), dpi=120)
ax.plot(Om_grid, chi_t - chi_t.min(), c='#D85A30', lw=2.2,
        label=f'Toward hemisphere (N={tw.sum()})')
ax.plot(Om_grid, chi_a - chi_a.min(), c='#185FA5', lw=2.2,
        label=f'Away hemisphere (N={aw.sum()})')
ax.axhline(1, c='k', lw=1.2, ls='--', label='1σ threshold (Δχ²=1)')
ax.axhline(4, c='k', lw=1.2, ls=':',  label='2σ threshold (Δχ²=4)')
ax.axvspan(0.28, 0.34, alpha=0.08, color='green')
ax.text(0.31, ax.get_ylim()[1] * 0.88 if ax.get_ylim()[1] > 5 else 8,
        'Planck\nprior', ha='center', fontsize=8, color='green')
ax.set_xlabel('Ωm', fontsize=12)
ax.set_ylabel('Δχ² from minimum', fontsize=12)
ax.set_title('Profile likelihood: H0-Ωm degeneracy at z < 0.15', fontsize=11)
ax.legend(fontsize=9)
ax.grid(alpha=0.2)
ax.set_xlim(0.10, 0.60)
plt.tight_layout()
plt.savefig('profile_likelihood_Om.png', dpi=150)
print("\nPlot saved: profile_likelihood_Om.png")
print("Done.")
