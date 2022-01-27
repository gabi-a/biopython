"""Todo: describe me."""

import fractions
import numpy as np
from Bio.PDB.xtal import SpaceGroup
from Bio.PDB.xtal.TransformType import TransformType


class CrystalTransform:
    """Representation of a transformation in a crystal.

    Representation of a transformation in a crystal:
    - a transformation id (each of the transformations in a space group, 0 to m)
    - a crystal translation
    The transformation matrix in crystal basis is stored, representing the basic
    transformation together with the crystal translation.
    Contains methods to check for equivalent transformations.
    """

    def __init__(self, sg=None, transform_id=0):
        """Return new CrystalTransform object.

        :param sg: Space group of the crystal transform. If None then transform_id must be 0 (Identity)
        :type sg: SpaceGroup

        :param transform_id: Transform id of the crystal transform.
        :type transform_id: int
        """
        if sg is None:
            if transform_id != 0:
                raise RuntimeError("Space Group cannot be None if transform_id != 0")
            else:
                self._mat_transform = np.eye(4)
        else:
            self._mat_transform = sg.get_transformation(transform_id).copy()

        self._space_group = sg
        self._transform_id = transform_id
        self._crystal_translation = np.array([0, 0, 0], dtype=int)

    def get_mat_transform(self):
        """Return matrix transform."""
        return self._mat_transform

    def set_mat_transform(self, m):
        """Set matrix transform."""
        self._mat_transform = m

    def get_crystal_translation(self):
        """Return crystal translation vector."""
        return self._crystal_translation

    def translate(self, translation):
        """Translate the crystal transform."""
        self.m[:3, 3] += translation
        self._crystal_translation += translation

    def is_equivalent(self, other):
        """Return true if the given CrystalTransform is equivalent to this one.

        Two crystal transforms are equivalent if one is the inverse of the other, i.e.
        their transformation matrices multiplication is equal to the identity.
        """
        mul = self.mat_transform @ other.mat_transform
        return np.allclose(mul, np.eye(4))

    def is_pure_crystal_translation(self):
        """Return True if pure crystal translation.

        Tells whether this transformation is a pure crystal lattice translation,
        i.e. no rotational component and an integer translation vector.
        """
        return self._transform_id == 0 and (
            self._crystal_translation[0] != 0
            or self._crystal_translation[1] != 0
            or self._crystal_translation[2] != 0
        )

    def is_identity(self):
        """Return True if identity transform."""
        return self._transform_id == 0 and np.allclose(self._crystal_translation, 0)

    def is_pure_translation(self):
        """Return True if pure translation.

        Tells whether this transformation is a pure translation:
        either a pure crystal (lattice) translation or a fractional (within
        unit cell) translation: space groups Ixxx, Cxxx, Fxxx have operators
        with fractional translations within the unit cell.
        """
        if self.is_pure_crystal_translation():
            return True
        if np.allclose(self._mat_transform[:3, :3], np.eye(3)) and np.any(
            np.abs(self._mat_transform[:3, 3]) > SpaceGroup.DELTA
        ):
            return True
        return False

    def is_fractional_translation(self):
        """Return True if fractional translation.

        Tells whether this transformation contains a fractional translational
        component (whatever its rotational component). A fractional translation
        together with a rotation means a screw axis.
        """
        return np.any(
            np.abs(self._mat_transform[:3, 3] - self._crystal_translation)
            > SpaceGroup.DELTA
        )

    def is_rotation(self):
        """Return True if rotation.

        Tells whether this transformation is a rotation disregarding the translational component,
        i.e. either pure rotation or screw rotation, but not improper rotation.
        """
        if self._space_group is None:
            return False

        return self._space_group.get_axis_fold_type(self._transform_id) > 1

    def get_transform_type(self):
        """Return the transform type.

        Returns the TransformType of this transformation: AU, crystal translation, fractional translation
        , 2 3 4 6-fold rotations, 2 3 4 6-fold screw rotations, -1 -3 -2 -4 -6 inversions/rotoinversions.
        """
        if self._space_group is None:
            return TransformType.AU

        fold_type = self._space_group.get_axis_fold_type(self._transform_id)
        is_screw_or_glide = False
        transl_screw_component = self.get_transl_screw_component()
        if np.any(np.abs(transl_screw_component) > SpaceGroup.DELTA):
            is_screw_or_glide = True

        if fold_type > 1:
            if is_screw_or_glide:
                if fold_type == 2:
                    return TransformType.TWOFOLDSCREW
                elif fold_type == 3:
                    return TransformType.THREEFOLDSCREW
                elif fold_type == 4:
                    return TransformType.FOURFOLDSCREW
                elif fold_type == 6:
                    return TransformType.SIXFOLDSCREW
                else:
                    raise RuntimeError(
                        "This transformation did not fall into any of the known types! This is most likely a bug."
                    )
            else:
                if fold_type == 2:
                    return TransformType.TWOFOLD
                elif fold_type == 3:
                    return TransformType.THREEFOLD
                elif fold_type == 4:
                    return TransformType.FOURFOLD
                elif fold_type == 6:
                    return TransformType.SIXFOLD
                else:
                    raise RuntimeError(
                        "This transformation did not fall into any of the known types! This is most likely a bug."
                    )
        elif fold_type < 0:
            if fold_type == -1:
                return TransformType.ONEBAR
            elif fold_type == -2:
                if is_screw_or_glide:
                    return TransformType.GLIDE
                return TransformType.TWOBAR
            elif fold_type == -3:
                return TransformType.THREEBAR
            elif fold_type == -4:
                return TransformType.FOURBAR
            elif fold_type == -6:
                return TransformType.SIXBAR
            else:
                raise RuntimeError(
                    "This transformation did not fall into any of the known types! This is most likely a bug."
                )
        else:
            if self.is_identity():
                return TransformType.AU
            if self.is_pure_crystal_translation():
                return TransformType.XTALTRANSL
            if self.is_fractional_translation():
                return TransformType.CELLTRANSL
            raise RuntimeError(
                "This transformation did not fall into any of the known types! This is most likely a bug."
            )

    def get_transform_id(self):
        """Return transform ID."""
        return self._transform_id

    def set_transform_id(self, transform_id):
        """Set transform ID."""
        self._transform_id = transform_id

    def __str__(self):
        return f"[{self._transform_id:2d}-({self.to_XYZ_string()})]"

    def to_XYZ_string(self):
        """Express this transformation in terms of x,y,z fractional coordinates."""
        out = ""
        for i in range(3):
            empty_row = True
            for j, coord in enumerate(["x", "y", "z"]):
                coef = self._mat_transform[i, j]
                if np.abs(coef) > 1e-6:
                    if np.abs(np.abs(coef) - 1) < 1e-6:
                        if coef < 0:
                            out += "-"
                        elif (j > 0) and not empty_row:
                            out += "+"
                    else:
                        if (j > 0) and (not empty_row) and (coef > 0):
                            out += "+"
                        out += self._format_coef(coef)
                        out += "*"
                    out += coord
                    empty_row = False
            # Intercept
            coef = self._mat_transform[i, 3]
            if np.abs(coef) > 1e-6:
                if (not empty_row) and (coef > 0):
                    out += "+"
                out += self._format_coef(coef)

            if i < 2:
                out += ","

        return out

    @staticmethod
    def _format_coef(coef):
        if np.abs(coef) < 1e-6:
            return "0"
        frac = fractions.Fraction(coef).limit_denominator(12)
        if np.abs(frac.numerator / frac.denominator - coef) < 1e-6:
            if frac.denominator == 1:
                return str(frac.numerator)
            return f"{frac.numerator}/{frac.denominator}"
        return f"{coef:.3f}"

    def get_transl_screw_component(self, m=None):
        """Return translation screw component."""
        if m is None:
            return self.get_transl_screw_component(self._mat_transform)

        fold_type = SpaceGroup.get_rot_axis_type(m)
        # For reference see:
        # http://www.crystallography.fr/mathcryst/pdf/Gargnano/Aroyo_Gargnano_1.pdf

        W = m[:3, :3].copy()

        if fold_type >= 0:
            # the Y matrix: Y = W^k-1 + W^k-2 ... + W + I  ; with k the fold type
            Y = np.eye(3)
            Wk = np.eye(3)
            for k in range(fold_type):
                Wk = Wk @ W
                if k != fold_type - 1:
                    Y = Y + Wk

            transl = m[:3, 3].copy()
            transl = Y @ transl
            transl /= fold_type
        else:
            if fold_type == -2:  # there are glide planes only in -2
                Y = np.eye(3) + W

                transl = m[:3, 3].copy()
                transl = Y @ transl
                transl /= 2.0
            else:  # for -1, -3, -4 and -6 there's nothing to do: fill with 0s
                transl = np.zeros(3)

        return transl
