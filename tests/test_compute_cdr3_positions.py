"""Tests for filter_utils.compute_cdr3_positions.

Run under the germinal conda env:

    source activate germinal
    python tests/test_compute_cdr3_positions.py

Targets the VL-first scFv regression that the old hardcoded
`cdr_lengths[0] + cdr_lengths[1]:` slice in run_structure_prediction's chai
fallback produced. The helper consolidates the 3-way branch so the bug cannot
re-appear in either run_filters or the chai fallback.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from germinal.filters.filter_utils import compute_cdr3_positions


class TestComputeCdr3Positions(unittest.TestCase):
    def test_vl_first_scfv_slices_last_cdr(self):
        """VL-first scFv: H3 is the last slot in cdr_positions."""
        cdr_lengths = [5, 7, 11, 5, 17, 15]  # L1,L2,L3,H1,H2,H3
        cdr_positions = list(range(sum(cdr_lengths)))
        self.assertEqual(
            compute_cdr3_positions({
                "type": "scfv",
                "vh_first": False,
                "cdr_positions": cdr_positions,
                "cdr_lengths": cdr_lengths,
            }),
            cdr_positions[45:],  # last 15
        )

    def test_vh_first_scfv_slices_middle_third(self):
        """VH-first scFv: H3 sits at [H1+H2 : H1+H2+H3]."""
        cdr_lengths = [5, 17, 15, 5, 7, 11]  # H1,H2,H3,L1,L2,L3
        cdr_positions = list(range(sum(cdr_lengths)))
        self.assertEqual(
            compute_cdr3_positions({
                "type": "scfv",
                "vh_first": True,
                "cdr_positions": cdr_positions,
                "cdr_lengths": cdr_lengths,
            }),
            cdr_positions[22:37],
        )

    def test_nb_slices_last_cdr(self):
        """Nanobody: flat CDR list, H3 is the last slot."""
        cdr_lengths = [7, 6, 17]
        cdr_positions = list(range(sum(cdr_lengths)))
        self.assertEqual(
            compute_cdr3_positions({
                "type": "nb",
                "cdr_positions": cdr_positions,
                "cdr_lengths": cdr_lengths,
            }),
            cdr_positions[13:],
        )

    def test_vh_first_default_true(self):
        """scFv without vh_first in config defaults to True."""
        cdr_lengths = [5, 17, 15, 5, 7, 11]
        cdr_positions = list(range(sum(cdr_lengths)))
        got = compute_cdr3_positions({
            "type": "scfv",
            "cdr_positions": cdr_positions,
            "cdr_lengths": cdr_lengths,
        })
        self.assertEqual(got, cdr_positions[22:37])

    def test_unknown_type_raises(self):
        with self.assertRaises(ValueError):
            compute_cdr3_positions({
                "type": "scab",
                "cdr_positions": [1, 2, 3],
                "cdr_lengths": [1, 1, 1],
            })


if __name__ == "__main__":
    unittest.main(verbosity=2)
