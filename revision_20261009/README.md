# MRM revision source snapshot, 2026-10-09

This is a local review and submission snapshot of the actual retained revision
drivers, their project-local dependencies, numerical modules, and small derived
configuration/result records. It contains no MRI k-space, image, sensitivity-map
arrays, patient identifiers, or participant-membership map. Data and saved maps
require separately authorized access. The matching repository snapshot commit is identified separately in the
submission code-provenance record.

## Snapshot identity

The cluster checkout is `/dataNAS/people/onat/mr_recon`, with Git base HEAD
`c87a60be6255a09d7167eb6aaad94159a1d94e90` and origin
`git@github.com:onat-dalmaz/fast_mri_gfactor.git` (upstream:
`https://github.com/danielabrahamgit/mr_recon.git`). That base commit is not an
exact publication revision: numerical modules and other tracked files are dirty,
and current revision drivers are untracked. `SNAPSHOT_MANIFEST.json` identifies
every included file by SHA256 and records original source hashes. No historical published commit matching these retained files was recovered. Included source files
were copied byte for byte; operators, reconstruction code, and results were not
changed. The environment files are the retained checkout declarations, not a
claim that all historical sessions used an identical environment. Current CS
timing records report Python 3.9.23, PyTorch 2.0.1, CUDA 11.7, and Quadro RTX 8000.

## Current experimental paths

* Cartesian: `scripts/cartesian_converged_target_audit.py`, 19 subjects, fixed
  zero-based partition 120 on axis 2, R=2, lambda=0, equal N=500, normal-operator
  scale 2, true residual tolerance 1e-5, and an analytical reference. The
  representative Figure 2 axis-0 slice is a separate 320 by 256 display.
* Physical spiral phantom: `scripts/run_noncart_final_grid.py` and its core
  modules use the unweighted pre-whitened operator at R=2/3/4 and fixed physical
  lambda=0.16338. Common pooled references contain 100000 samples at R=2 and
  60000 at R=3/4. The equal-N comparisons extend to 10000 samples per method.
  Certified solves use true residual tolerance 1e-5 and cap 1200. These reference
  budgets supersede the 30000-sample budget in historical demonstration files.
* In vivo spiral brain: `scripts/run_brain_noncart.py`,
  `scripts/brain_batch_reference_20260930.py`, and
  `scripts/pool_brain_reference.py` use R=2 and physical lambda=0.09145950233459473.
  The saved system entries give normalized lambda 0.003 accelerated and
  0.001499985782531369 full. Five 10000-sample batches form a 50000-sample pooled
  reference. Some CLI-argument fields in the archived PMR configuration retain
  older defaults; actual system entries govern. Brain time-to-target estimates
  use archived timings, with a separate same-session relative-cost check; that
  check does not establish identical execution sessions.
* Complex-TV CS: `scripts/run_cs_cohort_timing_rtx8000_20261007.py` is the current
  equal-N=2000 timing path. It processes 36 slices (13/17/21) from 12 volumes and
  11 participants. The cohort comprises 4 PD-fat-suppressed and 8 PD-without-fat-
  suppression volumes. It loads precomputed JSENSE maps from the Meddlr-processed
  local bundle (`jsense-12-cf=4/maps`), not maps provided by primary fastMRI.
  The original manuscript's 10/15 counts do not have a recovered historical
  manifest; no mapping between those counts and this documented rerun is claimed.

## CS estimator and execution scope

The current CS configuration uses TV lambda=3e-7, 30 FISTA outer steps, 50 TV-prox
steps, and beta_k=k/(k+3). PICO uses explicit cached tangent propagation, with
25 probes per batch; this timed path is not a generic automatic-differentiation
implementation. PMR executes sequential reconstructions and accumulates centered
Welford M2/N population variance. The one-time baseline reconstruction is counted in the timing.
Thus the observed roughly 1.84-fold timing ratio describes these implemented
pipelines, including their different batching/caching choices. Equal-N map
agreement is not accuracy against an independent converged reference.

Both CS paths report output standard deviation divided by input per-quadrature
sigma=1e-7. Real and imaginary probe signs are independent, with E|v|^2=2;
this normalization is consistent with the two input quadratures in PMR.
`scripts/run_cs_cohort_timing_20260930.py` is included because the current driver
imports it; its earlier timing result directory is superseded by the October 7
Quadro results. Do not substitute the earlier results for the current cohort.

Noise-level and TV-strength checks use equal N=3000 per method. The saved noise
sweep includes 200 sigma0 (see `metadata/cs_noise_level_metrics.json`). The
five-column display ends at 100 sigma0 solely to preserve the recorded layout;
`scripts/update_manifest_135.py` documents this display choice. No scientific
exclusion rationale was recovered. High-TV propagation describes the finite
30/50-step algorithm. The saved proximal fidelity check compares 50 versus 1000
prox steps on the final gradient-step input; it is not a full reconstruction
convergence comparison.

## Calibration and theoretical limits

`calibration_manifest.json` identifies the retained phantom and brain map files
by full-file SHA256, shape, and dtype. Phantom kernel 6 by 6 and SVD threshold
0.02 are archived manuscript-reported settings; the original calibration command,
central calibration extent, and crop threshold were not recovered. Brain kernel,
thresholds, calibration extent, and command were not recovered. The 36 processed
fastMRI map-slice content checksums are in
`metadata/fastMRI_map_slice_checksums.json`; these are C-order array-byte hashes,
not whole-HDF5 hashes. No saved map arrays are included.

Probe-variance optimality is restricted to independent proper complex probes
with fixed second moment: unit-modulus phase probes have fourth moment one and
remove the diagonal variance contribution under those assumptions. This is not
a claim of global optimality among improper, correlated, structured, or
covariance-aware probes. The non-Cartesian PICO numerator uses the numerical
Toeplitz normal, whereas PMR uses the actual forward/adjoint covariance. The
retained phantom check measures about 0.020% g-factor discrepancy for its tested
configuration; it is an empirical check, not a general error bound or a brain
verification.

## Reproduction and verification

`metadata/operating_points.json` and the experiment-specific manifests document
the current settings. `metadata/local_dependency_edges.json` records statically
resolved project-local imports. The drivers retain cluster data/result paths;
authorized inputs and an appropriate GPU environment are required before running
them. Historical preparation demos and original notebooks are not packaged as
current benchmark instructions. The snapshot was checked by parsing every
included Python source, reading every included JSON file, and verifying its
manifest checksums. No new reconstruction or noise experiments were run.
