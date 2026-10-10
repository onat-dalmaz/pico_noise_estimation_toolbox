# Fast Voxelwise SNR Estimation for Iterative MRI Reconstructions

PICO (Probing Image-space COvariance) estimates voxelwise noise variance for MRI reconstructions. This repository accompanies Dalmaz et al., *Fast Voxelwise SNR Estimation for Iterative MRI Reconstructions* (MRM-26-26969).

## Revision source snapshot

The [revision_20261009](revision_20261009/) directory contains the retained revision drivers, their numerical dependencies, configuration records, and a file-hash manifest. Its [README](revision_20261009/README.md) identifies the measured pipelines and required inputs. This snapshot supplements the original demonstration notebooks below; those notebooks retain their historical settings and are not a reproduction of all revised results.

The source snapshot contains no MRI or sensitivity-map arrays, patient identifiers, or participant-membership mapping. Full reruns require authorized source data and the archived coil maps. Map hashes identify the retained calibration outputs; they do not recover missing calibration-generation settings.

The revised linear comparisons use equal sample counts, a Cartesian analytical reference, and independent high-budget PMR references for the non-Cartesian cases. The physical phantom reference has 100,000 replicas at R=2 and 60,000 at R=3/4; the in vivo spiral brain reference has 50,000 replicas. Displayed linear maps are reciprocal 1/g, while quantitative NRMSE uses g. Both methods share their respective randomness between accelerated and fully sampled reconstructions.

The revised CS comparison uses equal N=2,000 across 36 slices from 12 fastMRI volumes and 11 participants. The timed PICO path uses cached manual tangent propagation through a finite 30-step FISTA-type reconstruction with 50-step complex-TV proximal solves. Its measured runtime describes that implementation; map agreement with PMR at equal N is not an independent true-covariance accuracy test. Noise-level and TV-strength sweeps use equal N=3,000.

## Estimator scope

For a linear reconstruction, PICO probes the image covariance directly without forming a dense covariance matrix. The stochastic diagonal identity is unbiased when the probe second moment is the identity. The random-phase variance argument concerns independent proper complex probes with unit second moment; it is not a global optimality claim over all probe families.

Nonquadratic regularized reconstructions use a local Jacobian model of noise propagation. The generic examples use automatic differentiation; the current timed CS experiment uses the explicit cached tangent path described above. The non-Cartesian implementation uses a Toeplitz normal-operator approximation. The archived phantom check found approximately 0.020% g-map discrepancy at its tested setting; that check is not a general error bound or a brain validation.

## Installation and original examples

The original standalone notebooks were assessed in the MR Reproducibility Study Group review. They provide representative single-slice examples with retained historical budgets.

```bash
conda env create -f environment.yml
conda activate mr_recon_env
pip install -e .
```

- [Cartesian knee notebook](notebooks/01_cartesian_knee.ipynb): CG-SENSE and analytical SENSE reference.
- [Non-Cartesian notebook](notebooks/02_noncartesian_spiral.ipynb): physical brain phantom with spiral sampling.
- [Compressed-sensing notebook](notebooks/03_compressed_sensing.ipynb): Jacobian propagation through TV reconstruction.

The original preparation and notebook-checking entry points remain `scripts/prepare_data.py` and `scripts/verify_notebooks.py`. The revision snapshot has separate drivers and retains cluster data/result paths; see its README before attempting full reruns. Source parsing and file-hash checks have been completed for the revision snapshot, but a clean-environment, end-to-end rerun has not been claimed.

## Data and code

Representative example bundles are retained under `experiments/`. Full multi-subject data are not redistributed.

- Cartesian knee: [mridata.org](https://mridata.org/).
- CS knee: [fastMRI](https://fastmri.med.nyu.edu/); the revised experiments load JSENSE maps from a separately processed Meddlr bundle, rather than from the primary fastMRI download.
- In vivo spiral brain: acquisition reported in [McCready et al., ISMRM 2025 abstract 3184](https://archive.ismrm.org/2025/3184_PB65iyQOS.html).

The toolbox builds on [mr_recon](https://github.com/danielabrahamgit/mr_recon). The revision directory preserves the source checkout's license and provenance. The Git base listed in the snapshot manifest is distinct from this repository's snapshot commit.

## Citation

Dalmaz O, Abraham DR, Toews AR, Chaudhari AS, Setsompop K, Hargreaves BA. *Fast Voxelwise SNR Estimation for Iterative MRI Reconstructions*. Manuscript MRM-26-26969.

Earlier workshop presentation: Dalmaz et al., *Fast SNR and g-factor mapping for image-based iterative reconstructions*, ISMRM Workshop on Data Sampling and Reconstruction, Sedona, 2026.

