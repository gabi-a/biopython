"""Unit tests for the Bio.PDB.xtal module."""
import unittest

try:
    import numpy as np
except ImportError:
    from Bio import MissingPythonDependencyError

    raise MissingPythonDependencyError(
        "Install NumPy if you want to use Bio.PDB."
    ) from None

from Bio.PDB.xtal import CrystalCell
from Bio.PDB.xtal import BravaisLattice
from Bio.PDB.vectors import Vector, rotaxis2m


class BravaisLatticeTests(unittest.TestCase):
    def test_triclinic(self):
        bl = BravaisLattice["TRICLINIC"]
        self.assertEqual(1, bl.get_id())
        self.assertEqual("TRICLINIC", bl.get_name())
        expected = CrystalCell(1.00, 1.25, 1.50, 60, 70, 80)
        result = bl.get_example_unit_cell()
        self.assertEqual(expected, result, f"Expected {expected}, got {result}")

        bl = BravaisLattice.TRICLINIC
        self.assertEqual(1, bl.get_id())
        self.assertEqual("TRICLINIC", bl.get_name())
        expected = CrystalCell(1.00, 1.25, 1.50, 60, 70, 80)
        result = bl.get_example_unit_cell()
        self.assertEqual(expected, result, f"Expected {expected}, got {result}")


class CrystalCellTests(unittest.TestCase):
    def test_get_cell_indices(self):
        cell = CrystalCell(100, 100, 100, 90, 90, 45)

        query = np.array([0, 0, 0])
        expected = np.array([0, 0, 0], dtype=int)
        result = cell.get_cell_indices(query)
        self.assertTrue(np.array_equal(expected, result), f"Wrong index for {query}")

        query = np.array([99.9, 0, 0])
        expected = np.array([0, 0, 0], dtype=int)
        result = cell.get_cell_indices(query)
        self.assertTrue(np.array_equal(expected, result), f"Wrong index for {query}")

        query = np.array([100, 0, 0])
        expected = np.array([1, 0, 0], dtype=int)
        result = cell.get_cell_indices(query)
        self.assertTrue(np.array_equal(expected, result), f"Wrong index for {query}")

        query = np.array([0, 50, 0])
        expected = np.array([-1, 0, 0], dtype=int)
        result = cell.get_cell_indices(query)
        self.assertTrue(np.array_equal(expected, result), f"Wrong index for {query}")

        query = np.array([51, 50, 0])
        expected = np.array([0, 0, 0], dtype=int)
        result = cell.get_cell_indices(query)
        self.assertTrue(np.array_equal(expected, result), f"Wrong index for {query}")

        query = np.array([72, 71, 0])
        expected = np.array([0, 1, 0], dtype=int)
        result = cell.get_cell_indices(query)
        self.assertTrue(np.array_equal(expected, result), f"Wrong index for {query}")

        query = np.array([500, 0, 0])
        expected = np.array([5, 0, 0], dtype=int)
        result = cell.get_cell_indices(query)
        self.assertTrue(np.array_equal(expected, result), f"Wrong index for {query}")

        query = np.array([-500, 0, 0])
        expected = np.array([-5, 0, 0], dtype=int)
        result = cell.get_cell_indices(query)
        self.assertTrue(np.array_equal(expected, result), f"Wrong index for {query}")

        query = np.array([-550, 0, 0])
        expected = np.array([-6, 0, 0], dtype=int)
        result = cell.get_cell_indices(query)
        self.assertTrue(np.array_equal(expected, result), f"Wrong index for {query}")

        query = np.array([2, 1, 500])
        expected = np.array([0, 0, 5], dtype=int)
        result = cell.get_cell_indices(query)
        self.assertTrue(np.array_equal(expected, result), f"Wrong index for {query}")

    def test_transform_to_origin(self):
        cell = CrystalCell(100, 100, 100, 90, 90, 45)
        h = 100 / 2 ** 0.5
        tol = 1e-6

        query = np.array([0, 0, 0])
        expected = np.array([0, 0, 0])
        result = cell.transf_to_origin_cell(query)
        self.assertTrue(
            np.allclose(expected, result, atol=tol),
            f"Error transforming to origin. Expected: {expected} but was: {result}",
        )

        query = np.array([99.9, 0, 0])
        expected = np.array([99.9, 0, 0])
        result = cell.transf_to_origin_cell(query)
        self.assertTrue(
            np.allclose(expected, result, atol=tol),
            f"Error transforming to origin. Expected: {expected} but was: {result}",
        )

        query = np.array([100, 0, 0])
        expected = np.array([0, 0, 0])
        result = cell.transf_to_origin_cell(query)
        self.assertTrue(
            np.allclose(expected, result, atol=tol),
            f"Error transforming to origin. Expected: {expected} but was: {result}",
        )

        query = np.array([0, 50, 0])
        expected = np.array([100, 50, 0])
        result = cell.transf_to_origin_cell(query)
        self.assertTrue(
            np.allclose(expected, result, atol=tol),
            f"Error transforming to origin. Expected: {expected} but was: {result}",
        )

        query = np.array([51, 50, 0])
        expected = np.array([51, 50, 0])
        result = cell.transf_to_origin_cell(query)
        self.assertTrue(
            np.allclose(expected, result, atol=tol),
            f"Error transforming to origin. Expected: {expected} but was: {result}",
        )

        query = np.array([h + 2, h + 1, 0])
        expected = np.array([2, 1, 0])
        result = cell.transf_to_origin_cell(query)
        self.assertTrue(
            np.allclose(expected, result, atol=tol),
            f"Error transforming to origin. Expected: {expected} but was: {result}",
        )

        query = np.array([500, 0, 0])
        expected = np.array([0, 0, 0])
        result = cell.transf_to_origin_cell(query)
        self.assertTrue(
            np.allclose(expected, result, atol=tol),
            f"Error transforming to origin. Expected: {expected} but was: {result}",
        )

        query = np.array([-500, 0, 0])
        expected = np.array([0, 0, 0])
        result = cell.transf_to_origin_cell(query)
        self.assertTrue(
            np.allclose(expected, result, atol=tol),
            f"Error transforming to origin. Expected: {expected} but was: {result}",
        )

        query = np.array([2, 1, 500])
        expected = np.array([2, 1, 0])
        result = cell.transf_to_origin_cell(query)
        self.assertTrue(
            np.allclose(expected, result, atol=tol),
            f"Error transforming to origin. Expected: {expected} but was: {result}",
        )

    def test_transform_to_origin_array(self):
        cell = CrystalCell(100, 100, 100, 90, 90, 45)
        h = 100 / 2 ** 0.5
        tol = 1e-6

        query = np.array([0, 0, 0])
        expected = np.array([0, 0, 0])
        result = cell.transf_ref_to_origin_cell([query], query)[0]
        self.assertTrue(
            np.allclose(expected, result, atol=tol),
            f"Error transforming to origin. Expected: {expected} but was: {query}",
        )

        query = np.array([99.9, 0, 0])
        expected = np.array([99.9, 0, 0])
        result = cell.transf_ref_to_origin_cell([query], query)[0]
        self.assertTrue(
            np.allclose(expected, result, atol=tol),
            f"Error transforming to origin. Expected: {expected} but was: {query}",
        )

        query = np.array([100, 0, 0])
        expected = np.array([0, 0, 0])
        result = cell.transf_ref_to_origin_cell([query], query)[0]
        self.assertTrue(
            np.allclose(expected, result, atol=tol),
            f"Error transforming to origin. Expected: {expected} but was: {query}",
        )

        query = np.array([0, 50, 0])
        expected = np.array([100, 50, 0])
        result = cell.transf_ref_to_origin_cell([query], query)[0]
        self.assertTrue(
            np.allclose(expected, result, atol=tol),
            f"Error transforming to origin. Expected: {expected} but was: {query}",
        )

        query = np.array([51, 50, 0])
        expected = np.array([51, 50, 0])
        result = cell.transf_ref_to_origin_cell([query], query)[0]
        self.assertTrue(
            np.allclose(expected, result, atol=tol),
            f"Error transforming to origin. Expected: {expected} but was: {query}",
        )

        query = np.array([h + 2, h + 1, 0])
        expected = np.array([2, 1, 0])
        result = cell.transf_ref_to_origin_cell([query], query)[0]
        self.assertTrue(
            np.allclose(expected, result, atol=tol),
            f"Error transforming to origin. Expected: {expected} but was: {query}",
        )

        query = np.array([500, 0, 0])
        expected = np.array([0, 0, 0])
        result = cell.transf_ref_to_origin_cell([query], query)[0]
        self.assertTrue(
            np.allclose(expected, result, atol=tol),
            f"Error transforming to origin. Expected: {expected} but was: {query}",
        )

        query = np.array([-500, 0, 0])
        expected = np.array([0, 0, 0])
        result = cell.transf_ref_to_origin_cell([query], query)[0]
        self.assertTrue(
            np.allclose(expected, result, atol=tol),
            f"Error transforming to origin. Expected: {expected} but was: {query}",
        )

        query = np.array([2, 1, 500])
        expected = np.array([2, 1, 0])
        result = result = cell.transf_ref_to_origin_cell([query], query)[0]
        self.assertTrue(
            np.allclose(expected, result, atol=tol),
            f"Error transforming to origin. Expected: {expected} but was: {query}",
        )

    def test_matrix_transf_to_origin_cell(self):
        cell = CrystalCell(100, 100, 100, 90, 90, 45)

        operations = []
        xtalOp = np.eye(4)

        # 90 deg rotation
        xtalOp[:3, :3] = rotaxis2m(np.pi / 2, Vector(0, 0, 1))
        operations.append(cell.transf_mat_to_orthonormal(xtalOp))

        # translate (+2,+1,-1) followed by 90 deg rotation
        xtalOp[:3, 3] = (2, 1, -1)
        operations.append(cell.transf_mat_to_orthonormal(xtalOp))

        xtalOp = np.eye(4)
        xtalOp[:3, :3] = rotaxis2m(-np.pi / 4, Vector(0, 0, 1))
        operations.append(xtalOp)

        # center of cell (.5,-2, 0)
        ref = np.array([50 - 3 * 25 * 2 ** 0.5, -3 * 25 * 2 ** 0.5, 50])

        transformed = cell.transf_ref_to_origin_cell_orthonormal(operations, ref)

        expected = np.zeros((3,), dtype=int)
        for op in transformed:
            x = op[:3, :3] @ ref + op[:3, 3]
            index = cell.get_cell_indices(x)
            self.assertTrue(
                np.array_equal(expected, index), f"Expected {expected}, got {index}"
            )
