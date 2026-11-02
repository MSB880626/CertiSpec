"""CertiSpec: distribution-free certification of positive and negative predictive values
for spectroscopic diagnosis from case-control data."""
from .core import (  # noqa: F401
    certify, certify_bands, certify_split, band_os, band_dkw, band_eval,
    os_band_levels, os_exact_coverage, risk_pos, risk_neg, run_data, run_external,
    validate, band_coverage,
)

__version__ = "1.0.0"
