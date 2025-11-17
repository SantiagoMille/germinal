# Germinal Containerization Bug Fixes Log

This document tracks bugs encountered during containerization and their fixes.

## BUG-001: Invalid matplotlib version in Dockerfile

**Date:** 2025-11-16

**Error:**
```
No solution found when resolving dependencies:
Because there is no version of matplotlib==3.8.5 and you require
matplotlib==3.8.5, we can conclude that the requirements are
unsatisfiable.
```

**Root Cause:**
Version `matplotlib==3.8.5` does not exist on PyPI. Available 3.8.x versions: 3.8.0, 3.8.1, 3.8.2, 3.8.3, 3.8.4

**Fix:**
Changed `matplotlib==3.8.5` to `matplotlib==3.8.4` in `Rayca-Code/Dockerfile` line 65

**Impact:**
No functional impact. matplotlib 3.8.4 is compatible with all Germinal requirements.

**Files Modified:**
- `Rayca-Code/Dockerfile`

## BUG-002: TypeError in metrics filtering phase (ipsae branch)

**Date:** 2025-11-17

**Error:**
```
TypeError: 'NoneType' object is not subscriptable
```

**Context:**
During full pipeline validation testing on A100 80GB GPU. Error occurs after:
- Hallucination phase completes (65 design iterations)
- PDB structures generated successfully
- Structure relaxation completes
- Structure prediction runs

**Root Cause:**
In `germinal/filters/filter_utils.py`, the `run_structure_prediction()` function (line 476) returns different values depending on the structure prediction model:
- **AF3 model** (lines 504-515): Returns `(external_pdb, external_metrics, ipsae)` with ipsae score
- **Chai model** (lines 516-532): Returns only `(external_pdb, external_metrics)`, leaving ipsae as `None` (default from line 503)

The `build_filter_metrics()` function then attempts to access `confidence_metrics["ipsae"]["ipsae"]` and `confidence_metrics["ipsae"]["pdockq2"]` without checking if ipsae is None, causing the TypeError.

**Fix Applied:**
Modified `germinal/filters/filter_utils.py` to add null safety checks:

**Line 287** (original):
```python
"ipsae": confidence_metrics["ipsae"]["ipsae"],
```

**Line 287** (fixed):
```python
"ipsae": confidence_metrics["ipsae"]["ipsae"] if confidence_metrics["ipsae"] is not None else None,
```

**Rationale:** When using Chai (instead of AF3) for structure prediction, ipsae scores are not computed. The conditional expression safely handles this by returning `None` when ipsae data is unavailable, rather than attempting to subscript a NoneType object.

**Line 333** (original):
```python
"ipsae_pdockq2": confidence_metrics["ipsae"]["pdockq2"],
```

**Line 333** (fixed):
```python
"ipsae_pdockq2": confidence_metrics["ipsae"]["pdockq2"] if confidence_metrics["ipsae"] is not None else None,
```

**Rationale:** Same issue - accessing pdockq2 from ipsae dictionary when ipsae is None. The fix provides consistency by setting both ipsae metrics to None when using Chai model.

**Impact:**
- Container infrastructure is NOT affected - all dependencies work correctly
- Fix enables pipeline completion for both AF3 and Chai structure prediction models
- ipsae and ipsae_pdockq2 metrics will be `None` when using Chai (expected behavior)

**Files Modified:**
- `germinal/filters/filter_utils.py` (lines 287, 333)

**Testing:**
- Rebuilt Docker image with fix (cached build: ~2 minutes)
- Ran full pipeline test: `max_trajectories=1 experiment_name=bug_fix_test`
- Result: Exit code 0 (success), no TypeError, completed in 5m 48s
- Pipeline correctly handled None ipsae values without crashing

**Validation Results After Fix:**
- Full pipeline completes without errors
- All stages work: Hallucination → Filtering → Structure Prediction → Metrics
- GPU memory usage: ~40GB on A100 80GB
- Pipeline runtime: ~6 minutes per trajectory
