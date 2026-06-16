# Cosmic Anisotropy — Pantheon+ Analysis

Code for:

**"Testing Axis Coherence Among Cosmic Anisotropy Probes: A Pantheon+ Analysis
of Hemispherical Expansion, Flow-Correction Artefacts, and Multi-Probe Dipole Structure"**

**Author:** Saksham Sharma (Independent Researcher, Noida, India)  
**Submitted to:** Physics of the Dark Universe  
**Preprint:** arXiv:astro-ph.CO (pending)

---

## Data

Download the Pantheon+ dataset before running any scripts:

```
git clone https://github.com/PantheonPlusSH0ES/DataRelease
```

You need two files:
- `Pantheon+SH0ES.dat` — distance moduli and redshifts
- `Pantheon+SH0ES_STAT+SYS.cov` — full covariance matrix

Place both in the same directory as the scripts, or update the file paths at the top of each script.

---

## Requirements

```
pip install numpy scipy pandas matplotlib
```

Python 3.9 or higher.

---

## Scripts

| Script | What it does |
|--------|-------------|
| `01_h0_hemisphere_scan.py` | H0 hemisphere comparison in zHD and zCMB frames with 10,000 shuffles |
| `02_tomographic_test.py` | H0 dipole amplitude across three redshift shells |
| `03_survey_jackknife.py` | Remove each contributing survey in turn and check signal stability |
| `04_profile_likelihood_Om.py` | Nonlinear H0–Ωm profile likelihood for both hemispheres |
| `05_coherence_test.py` | Multi-probe axis coherence test (4-probe and 3-probe) with Monte Carlo |

Run them in order. Each script saves its outputs as `.npy` files that the next script can optionally load.

---

## Random seeds

All Monte Carlo shuffle tests use `numpy.random.default_rng(42)` for reproducibility.
The coherence test uses `numpy.random.default_rng(2024)`.

---

## Citation

If you use this code, please cite:

Sharma S., 2026, "Testing Axis Coherence Among Cosmic Anisotropy Probes,"
Physics of the Dark Universe (submitted). arXiv:XXXX.XXXXX

And the Pantheon+ data release:

Scolnic D., et al., 2022, ApJ, 938, 113
