"""Describe the 17 transform types."""

from enum import Enum
from Bio.PDB.xtal import CrystalCell


class TransformType(Enum):
    """TransformType Enum."""

    #     id, fold,  screw, infinite, shortName
    AU = (0, 1, False, False, "AU")
    XTALTRANSL = (1, 1, False, True, "XT")  # translation
    CELLTRANSL = (2, 1, False, True, "FT")  # fractional translation

    TWOFOLD = (3, 2, False, False, "2")
    TWOFOLDSCREW = (4, 2, True, True, "2S")

    THREEFOLD = (5, 3, False, False, "3")
    THREEFOLDSCREW = (6, 3, True, True, "3S")

    FOURFOLD = (7, 4, False, False, "4")
    FOURFOLDSCREW = (8, 4, True, True, "4S")

    SIXFOLD = (9, 6, False, False, "6")
    SIXFOLDSCREW = (10, 6, True, True, "6S")

    ONEBAR = (11, -1, False, False, "-1")

    TWOBAR = (12, -2, False, False, "-2")
    GLIDE = (13, -2, True, False, "GL")

    THREEBAR = (14, -3, False, False, "-3")

    FOURBAR = (15, -4, False, False, "-4")

    SIXBAR = (16, -6, False, False, "-6")

    def get_id(self):
        """Return id."""
        return self.value[0]

    def get_fold_type(self):
        """Return fold type."""
        return self.value[1]

    def is_screw(self):
        """Return True if is screw type.

        Tells whether the transform type is a screw or glide plane.
        """
        return self.value[2]

    def is_infinite(self):
        """Return True if transform type produces infinite assemblies.

        Tells whether the transform type produces infinite assemblies
        if interface happens between identical chains
        """
        return self.value[3]

    def get_short_name(self):
        """Return short name."""
        return self.value[4]
