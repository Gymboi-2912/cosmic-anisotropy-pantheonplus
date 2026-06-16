"""
05_coherence_test.py
--------------------
Performs the multi-probe axis coherence test across four established
cosmic asymmetry probes.

Each dipole has a 180° sign ambiguity. The test optimises over all
2^N sign combinations to find the configuration that minimises the
mean angular spread from a common centre. Significance is assessed
against 10,000 random axis sets drawn uniformly on the sphere.

Two versions are run:
  - 4-probe: CMB dipole, matter dipole, galaxy spin, CMB cold spot
  - 3-probe: same minus the contested galaxy-spin probe

Reproduces Section 4 and Figure 5 of Sharma (2026).

Outputs:
    coherence_4probe.npy  —  {'obs_spread', 'null', 'p', 'best_axis'}
    coherence_3probe.npy  —  same
"""

import numpy as np
from itertools import product

# ── Probe directions (RA, DEC in degrees) ──────────────────────────────────────
# Dipole ambiguity: each probe can point toward or away from listed direction.
# The test searches all 2^N combinations for maximum coherence.
PROBES_4 = {
    'CMB kinematic dipole':    (307.8,  -6.9),   # Planck 2020
    'Matter number-count dipole': (167.0, -7.0),  # Secrest et al. 2022, 2025
    'Galaxy spin dipole':      ( 57.0, -10.0),    # Shamir 2021 (contested)
    'CMB cold spot':           ( 49.0, -19.0),    # Cruz et al. 2005
}
PROBES_3 = {k: v for k, v in PROBES_4.items() if k != 'Galaxy spin dipole'}

def to_vec(ra_deg, dec_deg):
    ra  = np.radians(ra_deg)
    dec = np.radians(dec_deg)
    return np.array([np.cos(dec)*np.cos(ra),
                     np.cos(dec)*np.sin(ra),
                     np.sin(dec)])

def angsep(v1, v2):
    return np.degrees(np.arccos(np.clip(v1 @ v2, -1, 1)))

def best_coherence(vecs):
    """Find sign combination minimising mean angular spread from common centre."""
    N = len(vecs)
    best_spread = 1e9
    best_signs  = None
    best_axis   = None
    for signs in product([1, -1], repeat=N):
        chosen = np.array([signs[i] * vecs[i] for i in range(N)])
        mv     = chosen.mean(axis=0)
        mv    /= np.linalg.norm(mv)
        seps   = np.array([angsep(chosen[i], mv) for i in range(N)])
        spread = seps.mean()
        if spread < best_spread:
            best_spread = spread
            best_signs  = signs
            best_axis   = mv.copy()
    return best_spread, best_axis, best_signs

def run_coherence_test(probes_dict, label, n_trials=10000, seed=2024):
    print(f"\n{'='*50}")
    print(f"  {label}")
    print(f"{'='*50}")

    names = list(probes_dict.keys())
    vecs  = np.array([to_vec(*v) for v in probes_dict.values()])
    N     = len(vecs)

    obs_spread, best_axis, best_signs = best_coherence(vecs)
    bra  = np.degrees(np.arctan2(best_axis[1], best_axis[0])) % 360
    bdec = np.degrees(np.arcsin(np.clip(best_axis[2], -1, 1)))
    print(f"  Best-fit coherence axis: RA={bra:.1f}°  DEC={bdec:+.1f}°")
    print(f"  Observed mean spread: {obs_spread:.2f}°")

    print(f"\n  Individual separations from coherence axis:")
    chosen = np.array([best_signs[i] * vecs[i] for i in range(N)])
    for i, name in enumerate(names):
        sep = angsep(chosen[i], best_axis)
        direction = '→' if best_signs[i] == 1 else '← (antipode)'
        print(f"    {name:35s}  {sep:5.1f}°  {direction}")

    # Pairwise separations
    print(f"\n  Pairwise angular separations (minimum of direct/antipodal):")
    for i in range(N):
        for j in range(i+1, N):
            sep     = angsep(vecs[i], vecs[j])
            sep_min = min(sep, 180 - sep)
            print(f"    {names[i][:25]:25s} ↔ {names[j][:25]:25s}: {sep_min:.1f}°")

    # Monte Carlo null test
    print(f"\n  Running {n_trials} Monte Carlo trials...")
    rng  = np.random.default_rng(seed)
    null = np.zeros(n_trials)
    sign_matrix = np.array(list(product([1, -1], repeat=N)), dtype=float)
    for t in range(n_trials):
        rv = rng.standard_normal((N, 3))
        rv /= np.linalg.norm(rv, axis=1, keepdims=True)
        chosen_all = sign_matrix[:, :, None] * rv[None, :, :]
        mv_all     = chosen_all.mean(axis=1)
        norms      = np.linalg.norm(mv_all, axis=1, keepdims=True)
        mv_all    /= norms
        dots       = np.clip(np.einsum('sni,si->sn', chosen_all, mv_all), -1, 1)
        spreads    = np.degrees(np.arccos(dots)).mean(axis=1)
        null[t]    = spreads.min()

    p = (np.sum(null <= obs_spread) + 1) / (n_trials + 1)
    print(f"\n  Results:")
    print(f"  Observed spread:   {obs_spread:.2f}°")
    print(f"  Null median:       {np.median(null):.2f}°")
    print(f"  Null 5th pct:      {np.percentile(null, 5):.2f}°")
    print(f"  p-value:           {p:.4f}")
    print(f"  Interpretation:    observed spread is {'more' if obs_spread < np.median(null) else 'less'}"
          f" coherent than {100*(1-p):.0f}% of random configurations")

    result = {
        'obs_spread': obs_spread,
        'best_axis':  best_axis,
        'best_axis_ra': bra,
        'best_axis_dec': bdec,
        'null': null,
        'p': p,
        'probe_names': names,
    }
    return result

# ── Run both tests ─────────────────────────────────────────────────────────────
r4 = run_coherence_test(PROBES_4, '4-PROBE TEST (all probes)')
r3 = run_coherence_test(PROBES_3, '3-PROBE TEST (galaxy spin excluded)')

np.save('coherence_4probe.npy', r4, allow_pickle=True)
np.save('coherence_3probe.npy', r3, allow_pickle=True)

# ── Plot ───────────────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), dpi=120)
for ax, r, label, col in zip(axes,
                               [r4, r3],
                               ['4-probe (all)', '3-probe (galaxy spin excluded)'],
                               ['#D85A30', '#185FA5']):
    ax.hist(r['null'], bins=40, color='#8F8F94', alpha=0.85,
            label=f'10,000 random sets')
    ax.axvline(r['obs_spread'], c=col, lw=2.5,
               label=f"Observed: {r['obs_spread']:.1f}°")
    ax.axvline(np.percentile(r['null'], 5), c='k', lw=1.5, ls='--',
               label='5th percentile')
    ax.set_xlabel('Minimum mean angular spread (degrees)', fontsize=10)
    ax.set_yticks([])
    ax.set_title(f'{label}\np = {r["p"]:.3f}', fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)

plt.suptitle('Axis coherence Monte Carlo — no single preferred direction found',
             fontsize=10)
plt.tight_layout()
plt.savefig('coherence_test.png', dpi=150)
print("\nPlot saved: coherence_test.png")
print("\nAll done. Summary:")
print(f"  4-probe coherence p = {r4['p']:.4f}")
print(f"  3-probe coherence p = {r3['p']:.4f}")
