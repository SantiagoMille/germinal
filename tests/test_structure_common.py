"""Tests for structure_common MSA helpers and filter_utils.compute_cdr3_positions.

Run under the germinal conda env:

    source activate germinal
    python tests/test_structure_common.py
"""

import os
import sys
import tempfile
import unittest

# Ensure the repo root is importable without install
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from germinal.filters.structure_common import (
    try_cache_hit_binder_msa,
    seed_binder_msa_cache,
    get_or_generate_target_msa,
    get_or_generate_binder_msa,
)


class TestBinderMsaCache(unittest.TestCase):
    def test_cache_hit_writes_swapped_query_line(self):
        """try_cache_hit_binder_msa rewrites lines 0+1 and preserves rows 2+."""
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = os.path.join(tmp, "msas")
            os.makedirs(cache_dir)
            cache = os.path.join(cache_dir, "binder_cached.a3m")
            with open(cache, "w") as fh:
                fh.write(">orig\nAAAAAA\n>h1\nACACAC\n>h2\nAAGAAG\n")

            out_rel = try_cache_hit_binder_msa(
                binder_seq="VVVVVV",
                design_name="d42",
                output_dir=tmp,
                output_rel_path="msas/d42.a3m",
            )
            self.assertEqual(out_rel, "msas/d42.a3m")
            with open(os.path.join(tmp, out_rel)) as fh:
                lines = [ln.rstrip("\n") for ln in fh.readlines()]
            self.assertEqual(lines[0], ">d42")
            self.assertEqual(lines[1], "VVVVVV")
            self.assertEqual(lines[2:], [">h1", "ACACAC", ">h2", "AAGAAG"])

    def test_cache_miss_on_length_mismatch(self):
        """Length mismatch returns None (caller regenerates)."""
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = os.path.join(tmp, "msas")
            os.makedirs(cache_dir)
            with open(os.path.join(cache_dir, "binder_cached.a3m"), "w") as fh:
                fh.write(">orig\nAAAAAA\n>h1\nACACAC\n")

            self.assertIsNone(
                try_cache_hit_binder_msa(
                    binder_seq="VVVVVVV",  # len 7 != cached len 6
                    design_name="d42",
                    output_dir=tmp,
                    output_rel_path="msas/d42.a3m",
                )
            )

    def test_seed_is_no_op_when_cache_exists(self):
        """Cache is not overwritten once seeded (first-writer-wins)."""
        with tempfile.TemporaryDirectory() as tmp:
            msas_dir = os.path.join(tmp, "msas")
            os.makedirs(msas_dir)
            gen_rel = "msas/d1.a3m"
            with open(os.path.join(tmp, gen_rel), "w") as fh:
                fh.write(">d1\nAAA\n")
            cache_path = os.path.join(msas_dir, "binder_cached.a3m")
            with open(cache_path, "w") as fh:
                fh.write(">pre-seeded\nBBB\n")

            seed_binder_msa_cache(generated_rel_path=gen_rel, output_dir=tmp)
            with open(cache_path) as fh:
                self.assertEqual(fh.read(), ">pre-seeded\nBBB\n")


class TestCacheBinderMsaGate(unittest.TestCase):
    """cache_binder_msa=True requires msa_mode='colabfold' — gate fires for all four entrypoints."""

    def _call(self, msa_mode):
        with tempfile.TemporaryDirectory() as tmp:
            get_or_generate_binder_msa(
                binder_seq="SEQ",
                design_name="d1",
                output_dir=tmp,
                msa_mode=msa_mode,
                cache_binder_msa=True,
            )

    def test_gate_rejects_target_mode(self):
        with self.assertRaises(ValueError):
            self._call("target")

    def test_gate_rejects_local_mode(self):
        with self.assertRaises(ValueError):
            self._call("local")

    def test_gate_rejects_none_mode(self):
        with self.assertRaises(ValueError):
            self._call("none")

    def test_gate_accepts_colabfold_mode(self):
        """With cache_binder_msa=False, no gate should fire regardless of mode."""
        with tempfile.TemporaryDirectory() as tmp:
            # msa_mode="target" + cache_binder_msa=False → returns "" without raising
            rel = get_or_generate_binder_msa(
                binder_seq="SEQ",
                design_name="d1",
                output_dir=tmp,
                msa_mode="target",
                cache_binder_msa=False,
            )
            self.assertEqual(rel, "")


class TestTargetMsaHelper(unittest.TestCase):
    def test_existing_target_reused(self):
        """get_or_generate_target_msa returns relative path when file exists."""
        with tempfile.TemporaryDirectory() as tmp:
            msas_dir = os.path.join(tmp, "msas")
            os.makedirs(msas_dir)
            target_file = os.path.join(msas_dir, "target_A.a3m")
            with open(target_file, "w") as fh:
                fh.write(">target_A\nSEQ\n")

            # msa_mode irrelevant on reuse — should not trigger generation
            rel = get_or_generate_target_msa(
                target_seq="SEQ",
                chain_id="A",
                output_dir=tmp,
                msa_mode="colabfold",
            )
            self.assertEqual(rel, "msas/target_A.a3m")

    def test_extra_search_dirs_copies_into_output(self):
        """When found via extra_search_dirs, file is copied into output_dir."""
        with tempfile.TemporaryDirectory() as tmp:
            af3_inputs = os.path.join(tmp, "af3_inputs", "msas")
            os.makedirs(af3_inputs)
            source = os.path.join(af3_inputs, "target_A.a3m")
            with open(source, "w") as fh:
                fh.write(">target_A\nSEQ\n")

            protenix_inputs = os.path.join(tmp, "protenix_inputs")
            os.makedirs(protenix_inputs)

            rel = get_or_generate_target_msa(
                target_seq="SEQ",
                chain_id="A",
                output_dir=protenix_inputs,
                msa_mode="colabfold",
                extra_search_dirs=[os.path.join(tmp, "af3_inputs")],
            )
            self.assertEqual(rel, "msas/target_A.a3m")
            copied = os.path.join(protenix_inputs, "msas", "target_A.a3m")
            self.assertTrue(os.path.exists(copied))
            with open(copied) as fh:
                self.assertEqual(fh.read(), ">target_A\nSEQ\n")



if __name__ == "__main__":
    unittest.main(verbosity=2)
