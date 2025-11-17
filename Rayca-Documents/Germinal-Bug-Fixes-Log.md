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
Code bug in ipsae branch metrics filtering phase. Likely attempting to access an attribute or index on a None value returned from structure prediction or metrics calculation.

**Impact:**
- Container infrastructure is NOT affected - all dependencies work correctly
- Hallucination, redesign, and structure prediction stages complete successfully
- Bug prevents final metrics calculation and filtering

**Workaround:**
Pipeline successfully generates antibody structures. Manual inspection of output PDBs is possible while bug is fixed.

**Files Affected:**
- Likely in `germinal/filters/` metrics calculation code
- Not a containerization issue - exists in ipsae branch codebase

**Validation Results Despite Bug:**
- Generated PDB structures: 157KB-306KB
- Relaxed structures created
- Structure predictions completed
- GPU memory usage: ~40GB on A100 80GB
