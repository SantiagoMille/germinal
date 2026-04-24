"""
Shared helpers for structure prediction backends (af3, protenix, chai).

Contains logic that is identical across backends so features added here
automatically apply to all of them:

- MSA generation via colabfold (local mmseqs2 or remote API)
- A3M post-processing (strip lowercase insertions for fixed-length rows)
- Binder MSA caching via query-line swap (length-matched rows)

Backends still own: input format construction specific to their binary,
subprocess invocation, output parsing.

Originated as MVP1 of docs/superpowers/plans/2026-04-24-structure-common-mvp.md
(hunter v1.5 revalidation exposed cache_binder_msa being a silent no-op for
protenix because the cache logic lived only in af3.py).
"""

import os
import shutil
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor, TimeoutError
from typing import Optional

from colabfold.colabfold import run_mmseqs2


DEFAULT_BINDER_CACHE_FILENAME = "msas/binder_cached.a3m"


def remove_a3m_insertions(a3m_path: str) -> None:
    """Strip lowercase insertion characters from an A3M file.

    After stripping, all rows have uniform length == query length, which is
    required for AF3/Protenix and is what makes the query-line swap cache
    correct (position k of every row aligns to position k of the query).
    """
    with open(a3m_path, "r") as fh:
        lines = fh.readlines()
    new_lines = []
    for line in lines:
        line = line.replace("\x00", "")
        if line.startswith("#") or line.startswith(">"):
            new_lines.append(line)
        else:
            new_lines.append("".join(c for c in line if not c.islower()))
    with open(a3m_path, "w") as fh:
        fh.writelines(new_lines)


def generate_local_msa(
    sequence,
    design_name,
    output_dir,
    msa_db_dir,
    use_gpu=False,
    use_gpu_server=False,
    use_metagenomic_db=False,
):
    """Generate an unpaired MSA via local colabfold_search (mmseqs2)."""
    if use_gpu_server:
        print("Starting GPU server...")
        gpu_server_dir = os.path.join(msa_db_dir, "colabfold_envdb_202108_db")
        uniref30_db_dir = os.path.join(msa_db_dir, "uniref30_2302_db")
        gpu_server_process = subprocess.Popen(
            [
                "mmseqs", "gpuserver", gpu_server_dir,
                "--max-seqs", "10000",
                "--db-load-mode", "0",
                "--prefilter-mode", "1",
            ],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        print("GPU server started at PID", gpu_server_process.pid)
        gpu_server_process.wait()
        uniref30_server_process = subprocess.Popen(
            [
                "mmseqs", "gpuserver", uniref30_db_dir,
                "--max-seqs", "10000",
                "--db-load-mode", "0",
                "--prefilter-mode", "1",
            ],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        print("Uniref30 server started at PID", uniref30_server_process.pid)
        uniref30_server_process.wait()

    with tempfile.TemporaryDirectory() as tmpdir:
        fasta_path = os.path.join(tmpdir, f"{design_name}.fasta")
        with open(fasta_path, "w") as fasta_file:
            fasta_file.write(f">{design_name}\n{sequence}\n")
        msa_out_dir = os.path.join(output_dir, "msas")
        os.makedirs(msa_out_dir, exist_ok=True)
        cmd = ["colabfold_search"]
        if use_gpu:
            cmd += ["--gpu", "1"]
        if use_gpu_server:
            cmd += ["--gpu-server", "1"]
        if not use_metagenomic_db:
            cmd += ["--use-env", "0"]
        cmd += [fasta_path, msa_db_dir, msa_out_dir]
        print(f"Running: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"colabfold_search failed for {design_name}: {e}. Falling back to no MSA.")
            return ""
        a3m_file = os.path.join(msa_out_dir, f"0.a3m")
        if os.path.exists(a3m_file):
            shutil.move(a3m_file, os.path.join(msa_out_dir, f"{design_name}.a3m"))
            a3m_file = os.path.join(msa_out_dir, f"{design_name}.a3m")
            remove_a3m_insertions(a3m_file)
            return os.path.relpath(a3m_file, output_dir)
        print(f"colabfold_search failed for {design_name}: MSA not found at {a3m_file}. Falling back to no MSA.")
        return ""


def generate_colabfold_msa(sequence, design_name, output_dir, use_metagenomic_db=False):
    """Generate unpaired MSA via the remote ColabFold API."""
    try:
        print(f"Running colabfold_search for {design_name} with use_env={use_metagenomic_db}")
        run_mmseqs2(
            sequence,
            os.path.join(output_dir, f"{design_name}"),
            use_env=use_metagenomic_db,
        )
        print(f"colabfold_search finished for {design_name}")
    except Exception as e:
        print(f"colabfold_search failed for {design_name}: {e}. Falling back to no MSA.")
        return ""

    old_msa_path = os.path.join(output_dir, f"{design_name}_all", "uniref.a3m")
    if not os.path.exists(old_msa_path):
        print(f"colabfold_search failed for {design_name}: MSA not found at {old_msa_path}. Falling back to no MSA.")
        return ""
    remove_a3m_insertions(old_msa_path)
    new_msa_path = os.path.join(output_dir, f"msas/{design_name}.a3m")
    os.makedirs(os.path.dirname(new_msa_path), exist_ok=True)
    shutil.copyfile(old_msa_path, new_msa_path)
    shutil.rmtree(os.path.join(output_dir, f"{design_name}_all"))
    return os.path.relpath(new_msa_path, output_dir)


def call_generate_colabfold_msa_with_timeout(
    sequence, design_name, output_dir, timeout=120, use_metagenomic_db=False
):
    """Timeout-protected wrapper around generate_colabfold_msa."""
    with ProcessPoolExecutor(max_workers=1) as exe:
        fut = exe.submit(
            generate_colabfold_msa,
            sequence,
            design_name,
            output_dir,
            use_metagenomic_db,
        )
        try:
            return fut.result(timeout=timeout)
        except TimeoutError:
            fut.cancel()
            exe.shutdown(wait=False, cancel_futures=True)
            print(f"colabfold_search failed for {design_name}: timed out after {timeout}s. Returning empty MSA.")
            return ""


def _atomic_write_copy(src_path: str, dst_path: str) -> None:
    """Copy ``src_path`` to ``dst_path`` atomically on POSIX.

    Writes to a tmp file in the same directory, then ``os.replace`` (atomic
    within a filesystem). Prevents partial-write corruption when concurrent
    jobs with the same experiment_name touch the same cache file. Readers
    see either old or new complete content, never half.
    """
    dst_dir = os.path.dirname(dst_path)
    os.makedirs(dst_dir, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        prefix=".", suffix=".tmp", dir=dst_dir
    )
    os.close(fd)
    try:
        shutil.copy(src_path, tmp_path)
        os.replace(tmp_path, dst_path)
    except Exception:
        if os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
        raise


def _atomic_write_lines(lines, dst_path: str) -> None:
    """Write ``lines`` to ``dst_path`` atomically (see ``_atomic_write_copy``)."""
    dst_dir = os.path.dirname(dst_path)
    os.makedirs(dst_dir, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        prefix=".", suffix=".tmp", dir=dst_dir
    )
    try:
        with os.fdopen(fd, "w") as fh:
            fh.writelines(lines)
        os.replace(tmp_path, dst_path)
    except Exception:
        if os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
        raise


def try_cache_hit_binder_msa(
    binder_seq: str,
    design_name: str,
    output_dir: str,
    output_rel_path: str,
    cache_rel_path: str = DEFAULT_BINDER_CACHE_FILENAME,
    cache_abs_path: Optional[str] = None,
) -> Optional[str]:
    """Attempt a binder-MSA cache hit via query-line swap.

    If the cache (see below) exists and its query row length equals
    ``len(binder_seq)``, write a new a3m at ``output_dir/output_rel_path``
    where rows 2+ are copied from cache but row 0 (header) and row 1 (query)
    are rewritten for this ``design_name`` and ``binder_seq``.

    Works because ``remove_a3m_insertions`` strips lowercase insertions so all
    rows have uniform length == query length; swapping line 1 keeps position
    k in every row aligned to position k of the query.

    Cache location:
        - If ``cache_abs_path`` is provided, it is used verbatim (absolute
          path). This lets callers point the cache at a stable location
          outside ``output_dir`` — e.g., AF3's batch path uses a PID-scoped
          output_dir but needs the cache to live at a stable parent so it
          persists across runs.
        - Otherwise the cache is at ``output_dir/cache_rel_path``.

    Concurrency: the output a3m is written via atomic rename so concurrent
    writers (e.g., multiple jobs sharing ``experiment_name``) do not see
    partial files. Cache reads tolerate short/partial caches by falling
    through to regeneration.

    Returns the relative path ``output_rel_path`` on hit, or ``None`` on miss
    (caller should generate the MSA fresh).
    """
    cache_path = cache_abs_path if cache_abs_path is not None else os.path.join(output_dir, cache_rel_path)
    if not os.path.exists(cache_path):
        return None
    try:
        with open(cache_path) as fh:
            cached_lines = fh.readlines()
    except OSError:
        return None
    if len(cached_lines) < 2:
        return None
    cached_query = cached_lines[1].strip()
    if len(cached_query) != len(binder_seq):
        return None
    new_lines = list(cached_lines)
    new_lines[0] = f">{design_name}\n"
    new_lines[1] = f"{binder_seq.upper()}\n"
    full_path = os.path.join(output_dir, output_rel_path)
    _atomic_write_lines(new_lines, full_path)
    return output_rel_path


def get_or_generate_binder_msa(
    binder_seq: str,
    design_name: str,
    output_dir: str,
    msa_mode: str,
    cache_binder_msa: bool = False,
    use_metagenomic_db: bool = False,
    msa_db_dir: Optional[str] = None,
    output_rel_path: Optional[str] = None,
    cache_abs_path: Optional[str] = None,
) -> str:
    """Unified binder-MSA acquisition for all structure-prediction backends.

    Resolution order:
      1. If ``cache_binder_msa`` is True, try a cross-design cache hit
         (length-matched query-line swap from ``cache_abs_path`` or the
         default ``<output_dir>/msas/binder_cached.a3m``).
      2. If ``msa_mode == "target"``: return "" (skip binder MSA — caller
         decides what to do, e.g., single-seq fallback).
      3. Otherwise generate fresh via local colabfold_search
         (``msa_mode == "local"``) or the remote ColabFold API
         (``msa_mode in ("colabfold", "target")``).
      4. If ``cache_binder_msa`` is True and step 3 produced an MSA, seed
         the cache so subsequent candidates can hit it.

    Returns the relative path (relative to ``output_dir``) of the MSA
    written for this binder, or ``""`` if generation was skipped or failed.

    Using one function across af3 / protenix / (future) chai guarantees
    identical cache semantics everywhere — any fix applied here propagates.

    Args:
        binder_seq: Binder amino acid sequence.
        design_name: Unique identifier; determines the on-disk output
            filename ``msas/{design_name}.a3m``. Override via
            ``output_rel_path`` if caller wants a different name.
        output_dir: Where generated/hit MSA files are written. Safe to be
            PID-scoped — the cache lives at ``cache_abs_path`` (caller's
            choice of stable location), independent of ``output_dir``.
        msa_mode: ``"local"``, ``"colabfold"``, ``"target"``.
        cache_binder_msa: Enable the cross-design cache.
        use_metagenomic_db: Pass-through to MSA generation.
        msa_db_dir: Required when ``msa_mode == "local"``.
        output_rel_path: Override the default ``msas/{design_name}.a3m``.
        cache_abs_path: Absolute path to the shared binder cache file.
            Defaults to ``<output_dir>/msas/binder_cached.a3m``. Callers
            using PID-scoped ``output_dir``s should pass a stable path so
            the cache persists across runs.
    """
    if output_rel_path is None:
        output_rel_path = os.path.join("msas", f"{design_name}.a3m")

    # Gate: cache_binder_msa relies on a real binder MSA being generated first
    # (the first binder seeds the cache; subsequent ones rewrite only the query
    # line). With msa_mode="target" no binder MSA is ever generated, so the
    # cache cannot seed and the flag silently does nothing. Fail loud instead.
    # Local mode would work in theory but is untested; require colabfold.
    if cache_binder_msa and msa_mode != "colabfold":
        raise ValueError(
            f"cache_binder_msa=True requires msa_mode='colabfold'; got "
            f"msa_mode={msa_mode!r}. In any other mode the binder MSA is "
            f"either skipped (target) or untested (local/none), so the cache "
            f"cannot seed and the flag silently does nothing. Set "
            f"cache_binder_msa=false or switch msa_mode to 'colabfold'."
        )

    # Step 1: attempt cache hit
    if cache_binder_msa:
        hit = try_cache_hit_binder_msa(
            binder_seq=binder_seq,
            design_name=design_name,
            output_dir=output_dir,
            output_rel_path=output_rel_path,
            cache_abs_path=cache_abs_path,
        )
        if hit is not None:
            return hit

    # Step 2: target-only mode skips binder MSA
    if msa_mode == "target":
        return ""

    # Step 3: fresh generation
    if msa_mode == "local":
        if msa_db_dir is None:
            raise ValueError("msa_mode='local' requires msa_db_dir")
        rel_path = generate_local_msa(
            sequence=binder_seq,
            design_name=design_name,
            output_dir=output_dir,
            msa_db_dir=msa_db_dir,
            use_metagenomic_db=use_metagenomic_db,
        )
    elif msa_mode == "colabfold":
        rel_path = call_generate_colabfold_msa_with_timeout(
            sequence=binder_seq,
            design_name=design_name,
            output_dir=output_dir,
            use_metagenomic_db=use_metagenomic_db,
        )
    else:
        return ""

    # Step 4: seed cache on first successful generation
    if cache_binder_msa and rel_path:
        seed_binder_msa_cache(
            generated_rel_path=rel_path,
            output_dir=output_dir,
            cache_abs_path=cache_abs_path,
        )
    return rel_path


def get_or_generate_target_msa(
    target_seq: str,
    chain_id: str,
    output_dir: str,
    msa_mode: str,
    use_metagenomic_db: bool = False,
    msa_db_dir: Optional[str] = None,
    extra_search_dirs: Optional[list] = None,
) -> str:
    """Unified target-chain MSA acquisition for all structure-prediction backends.

    Resolution order:
      1. Reuse an existing ``<output_dir>/msas/target_{chain_id}.a3m``.
      2. If ``extra_search_dirs`` is given, look for the same relative path under
         each; on hit, copy that file into ``output_dir`` (so downstream code
         that consumes the path relative to ``output_dir`` resolves correctly
         regardless of which base_dir owned the original).
      3. Otherwise generate fresh via ``generate_local_msa``
         (``msa_mode == "local"``) or
         ``call_generate_colabfold_msa_with_timeout``
         (``msa_mode in ("colabfold", "target")``).

    Returns the relative path (from ``output_dir``) on success, or ``""`` when
    no MSA was produced (unknown mode or generation failed).
    """
    design_name = f"target_{chain_id}"
    rel_path = os.path.join("msas", f"{design_name}.a3m")
    full_path = os.path.join(output_dir, rel_path)

    if os.path.exists(full_path):
        return rel_path

    if extra_search_dirs:
        for extra in extra_search_dirs:
            candidate = os.path.join(extra, rel_path)
            if os.path.exists(candidate):
                os.makedirs(os.path.dirname(full_path), exist_ok=True)
                _atomic_write_copy(candidate, full_path)
                return rel_path

    if msa_mode == "local":
        if msa_db_dir is None:
            raise ValueError("msa_mode='local' requires msa_db_dir")
        return generate_local_msa(
            sequence=target_seq,
            design_name=design_name,
            output_dir=output_dir,
            msa_db_dir=msa_db_dir,
            use_metagenomic_db=use_metagenomic_db,
        )
    if msa_mode in ("colabfold", "target"):
        return call_generate_colabfold_msa_with_timeout(
            sequence=target_seq,
            design_name=design_name,
            output_dir=output_dir,
            use_metagenomic_db=use_metagenomic_db,
        )
    return ""


def seed_binder_msa_cache(
    generated_rel_path: str,
    output_dir: str,
    cache_rel_path: str = DEFAULT_BINDER_CACHE_FILENAME,
    cache_abs_path: Optional[str] = None,
) -> None:
    """Seed the binder MSA cache with a freshly-generated MSA.

    No-op if the cache already exists. Called by backends right after they
    generate the first binder MSA for a run; subsequent candidates hit the
    cache via ``try_cache_hit_binder_msa``.

    Cache location: see docstring on ``try_cache_hit_binder_msa``. Pass the
    same ``cache_abs_path`` here as there.

    Concurrency: cache file is created via atomic rename, so concurrent
    seeds (e.g., two jobs sharing ``experiment_name``) produce last-writer-
    wins semantics without partial-file corruption.
    """
    cache_path = cache_abs_path if cache_abs_path is not None else os.path.join(output_dir, cache_rel_path)
    if os.path.exists(cache_path):
        return
    full_generated = os.path.join(output_dir, generated_rel_path)
    if not os.path.exists(full_generated):
        return
    _atomic_write_copy(full_generated, cache_path)
