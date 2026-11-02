# CertiSpec

**Distribution-free certification of positive and negative predictive values for spectroscopic
diagnosis from case-control data.**

CertiSpec turns a spectroscopic classifier into a decision procedure with explicit,
finite-sample guarantees. For a maximum tolerated risk α, a confidence level 1 − δ and a target
prevalence π chosen by the user, it returns thresholds λ− < λ+ such that, with probability at
least 1 − δ,

- 1 − NPV of the patients declared negative (score ≤ λ−) is at most α, and
- 1 − PPV of the patients declared positive (score > λ+) is at most α,

while patients in between are referred for confirmatory testing. When the calibration data are
insufficient, it abstains and reports the smallest risk the data can certify.

## Key properties

- **Case-control data and any prevalence.** Bounds are placed on the class-conditional score
  distributions, so the guarantee holds for every target prevalence, independently of the
  proportion of cases in the study.
- **Simultaneous over thresholds and prevalences.** Thresholds and prevalence can be chosen
  after inspecting the calibration data without loss of validity.
- **Exact order-statistic bands.** Band coverage is computed exactly (not approximated) and is
  valid for continuous, discrete and tied scores.
- **Patient level.** Replicate spectra are averaged per patient; partitions never split a patient.
- **Attribution control.** Optional refit on spectral regions without sample absorption bands
  flags certifications that are not confined to absorption bands.
- Comparators: Dvoretzky-Kiefer-Wolfowitz bands and split-calibration Learn then Test.

The theory, proofs and validation are described in the accompanying article (see *Citation*).

## Installation

```bash
git clone https://github.com/MSB880626/CertiSpec.git
cd CertiSpec
pip install -e ".[test,scripts]"
pytest -q            # 123 tests, about 1-2 minutes
```

Requires Python ≥ 3.9, NumPy, SciPy, pandas and scikit-learn.

## Usage

Input: a CSV file with one row per spectrum and columns `patient_id`, `label` (0/1) and one
column per absorbance (headers = wavenumbers).

```bash
# Repeated patient-level partitions (training / calibration / test)
certispec data cohort.csv --reps 50 --train 0.5 --cal 0.3 \
    --wn-ranges 900-1800,2800-3700 --null-ranges 2400-2700,3800-4000

# Fixed external test set
certispec external train.csv test.csv --reps 50 --cal 0.6

# Simulation study and band coverage
certispec validate --reps 2000
certispec coverage --reps 20000
```

Each analysis writes a summary table and a per-partition detail file with certified
thresholds, fractions of positive, negative and undecided patients, certified and smallest
certifiable risks, the attribution flag, and (with test patients) simultaneous 95%
confidence intervals.

From Python:

```python
import numpy as np
from certispec import certify

res = certify(scores_negative, scores_positive, "os", pis=[0.10], alphas=[0.05],
              rng=np.random.default_rng(0))
```

See `examples/quickstart.py`.

## Assumptions

(A1) Calibration patients are independent within each class and disjoint from the patients used
to fit the classifier. (A2) The class-conditional score distributions transfer to the target
population. (A2) cannot be verified from calibration data; external validation is required.

## Reproducing the article

```bash
python scripts/reproduce_cohorts_and_simulations.py   # cohorts, attribution, coverage, simulations
python scripts/reproduce_positive_control.py          # semisynthetic positive control
```

See `data/README.md` for obtaining the datasets.

## Citation

If you use CertiSpec, please cite the article and the software (see `CITATION.cff`):

> Sánchez Brito M. Distribution-free certification of predictive values for infrared
> spectroscopic diagnosis from case–control cohorts. (in preparation)

> Sánchez Brito M. CertiSpec (version 1.0.0) [software]. Zenodo. doi: ZENODO_DOI

## License

MIT License. See `LICENSE`.
