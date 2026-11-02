# Data

CertiSpec does not redistribute third-party spectra. The three public cohorts analysed in the
article can be obtained as follows.

| Cohort | Source | Preparation |
|---|---|---|
| Saliva, type 2 diabetes vs controls (1040 patients) | Sánchez-Brito et al., Comput Struct Biotechnol J 2022, doi:10.1016/j.csbj.2022.08.038 | Columns `label` (DM2/Control) and absorbances; converted inside `scripts/reproduce_cohorts_and_simulations.py` |
| Plasma, severe vs non-severe COVID-19 | Zenodo record 4805258 (https://zenodo.org/records/4805258); Banerjee et al., Anal Chem 2021 | `scripts/prepare_plasma.py` with `plasma_clinical.csv` |
| Serum, schistosomiasis vs controls | https://github.com/victorffaguilar10/ATR-FTIR-Schistosomiasis | `General_and_truncated_spectra.xlsx` and `Localidade_truncado.xlsx` |

`plasma_clinical.csv` transcribes the clinical tables S1-S4 of the supporting information of the
plasma study (plasma identifier, hospital registration number, age, sex, comorbidities, severity
and set). The registration number is used as the patient identifier.
