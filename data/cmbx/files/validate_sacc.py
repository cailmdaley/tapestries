"""Validate SACC file completeness and structure.

Checks:
- All expected tracers present (NZTracer for shear, MiscTracer for CMB kappa)
- n(z) arrays have expected shape and normalization
- Covariance matrix present and positive semi-definite
- Data types and counts match expectations
- Outputs a JSON validation report
"""

import json
from pathlib import Path

import numpy as np
import sacc

from dr1_notebooks.scratch.cdaley.snakemake_helpers import snakemake_log


snakemake = snakemake  # type: ignore # noqa: F821

sacc_path = Path(snakemake.input.sacc)
output_report = Path(snakemake.output.report)

snakemake_log(snakemake, f"Validating SACC file: {sacc_path}")

s = sacc.Sacc.load_fits(str(sacc_path))

checks = {}
all_pass = True


def check(name, condition, detail=""):
    global all_pass
    checks[name] = {"pass": bool(condition), "detail": str(detail)}
    status = "PASS" if condition else "FAIL"
    snakemake_log(snakemake, f"  [{status}] {name}: {detail}")
    if not condition:
        all_pass = False


# --- Tracer checks ---
n_tracers = len(s.tracers)
check("has_tracers", n_tracers > 0, f"{n_tracers} tracers")

nz_tracers = {n: t for n, t in s.tracers.items() if isinstance(t, sacc.tracers.NZTracer)}
misc_tracers = {n: t for n, t in s.tracers.items() if isinstance(t, sacc.tracers.MiscTracer)}

check("shear_tracers_are_nz", len(nz_tracers) >= 14,
      f"{len(nz_tracers)} NZTracer (expect >=14: 2 methods × 7 bins)")

check("cmb_kappa_is_misc", len(misc_tracers) >= 1,
      f"{len(misc_tracers)} MiscTracer (expect >=1: CMB kappa)")

# --- n(z) checks ---
for name, t in nz_tracers.items():
    check(f"nz_{name}_z_length", len(t.z) == 3000, f"z has {len(t.z)} points")
    check(f"nz_{name}_nz_positive", t.nz.sum() > 0, f"nz sum = {t.nz.sum():.4f}")
    check(f"nz_{name}_z_range", t.z[0] == 0.0 and np.isclose(t.z[-1], 6.0, atol=0.01),
          f"z = [{t.z[0]:.1f}, {t.z[-1]:.1f}]")

# --- Data checks ---
n_data = len(s.data)
data_types = set(d.data_type for d in s.data)

check("has_data", n_data > 0, f"{n_data} data points")
check("has_cl_0e", "cl_0e" in data_types, f"data types: {data_types}")
check("has_cl_0b", "cl_0b" in data_types, "B-mode null test present")

n_ells_per_spectrum = n_data // (len(nz_tracers) * 2)  # 2 data types per tracer pair
check("data_count_consistent", n_data == len(nz_tracers) * 2 * n_ells_per_spectrum,
      f"{n_data} = {len(nz_tracers)} tracers × 2 types × {n_ells_per_spectrum} ells")

# --- Covariance checks ---
has_cov = s.covariance is not None
check("has_covariance", has_cov, "covariance matrix present" if has_cov else "NO covariance")

if has_cov:
    cov = s.covariance.covmat
    check("cov_shape", cov.shape == (n_data, n_data),
          f"shape {cov.shape} (expect {n_data}×{n_data})")
    check("cov_diagonal_positive", np.all(np.diag(cov) >= 0),
          f"min diag = {np.diag(cov).min():.2e}")
    check("cov_symmetric", np.allclose(cov, cov.T),
          f"max asymmetry = {np.abs(cov - cov.T).max():.2e}")

    eigvals = np.linalg.eigvalsh(cov)
    check("cov_positive_semidefinite", np.all(eigvals >= -1e-15),
          f"min eigenvalue = {eigvals.min():.2e}")

# --- Summary ---
n_pass = sum(1 for c in checks.values() if c["pass"])
n_total = len(checks)
check("all_checks_pass", all_pass, f"{n_pass}/{n_total} checks passed")

report = {
    "sacc_file": str(sacc_path),
    "n_tracers": n_tracers,
    "n_nz_tracers": len(nz_tracers),
    "n_data_points": n_data,
    "data_types": sorted(data_types),
    "has_covariance": has_cov,
    "all_pass": all_pass,
    "checks": checks,
}

output_report.parent.mkdir(parents=True, exist_ok=True)
with open(output_report, "w") as f:
    json.dump(report, f, indent=2)

snakemake_log(snakemake, f"\nValidation {'PASSED' if all_pass else 'FAILED'}: {n_pass}/{n_total} checks")
snakemake_log(snakemake, f"Report: {output_report}")
