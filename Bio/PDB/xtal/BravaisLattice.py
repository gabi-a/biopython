"""Describes the 7 Bravais Lattice types."""

from enum import Enum
from Bio.PDB.xtal import CrystalCell


class BravaisLattice(Enum):
    """Describe the 7 Bravais Lattice types."""

    TRICLINIC = (
        1,
        "TRICLINIC",
        CrystalCell(1.00, 1.25, 1.50, 60, 70, 80),
    )  # alpha,beta,gamma!=90
    MONOCLINIC = (
        2,
        "MONOCLINIC",
        CrystalCell(1.00, 1.25, 1.50, 90, 60, 90),
    )  # beta!=90, alpha=gamma=90
    ORTHORHOMBIC = (
        3,
        "ORTHORHOMBIC",
        CrystalCell(1.00, 1.25, 1.50, 90, 90, 90),
    )  # alpha=beta=gamma=90
    TETRAGONAL = (
        4,
        "TETRAGONAL",
        CrystalCell(1.00, 1.00, 1.25, 90, 90, 90),
    )  # alpha=beta=gamma=90, a=b
    TRIGONAL = (
        5,
        "TRIGONAL",
        CrystalCell(1.00, 1.00, 1.25, 90, 90, 120),
    )  # a=b!=c, alpha=beta=90, gamma=120
    HEXAGONAL = (
        6,
        "HEXAGONAL",
        CrystalCell(1.00, 1.00, 1.25, 90, 90, 120),
    )  # a=b!=c, alpha=beta=90, gamma=120
    CUBIC = (
        7,
        "CUBIC",
        CrystalCell(1.00, 1.00, 1.00, 90, 90, 90),
    )  # a=b=c, alpha=beta=gamma=90

    def get_id(self):
        """Return id."""
        return self.value[0]

    def get_name(self):
        """Return name."""
        return self.value[1]

    def get_example_unit_cell(self):
        """Return example unit cell."""
        return self.value[2]
