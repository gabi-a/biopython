"""Class to handle space groups."""

import re
import os
import warnings
import fractions
import xml.etree.ElementTree as ET

import numpy as np

from Bio.PDB.xtal import BravaisLattice
from Bio import BiopythonWarning
from Bio.PDB.vectors import Vector, rotaxis2m, m2rotaxis


class SymoplibParser:
    """Static class to load space groups."""

    _sgs = None
    _name2id = None
    _name_pat = re.compile(".*\\s([A-Z]+)(\\s'.+')?\\s+'(.+)'.*")

    @classmethod
    def get_space_groups(cls):
        """Return list of space groups."""
        if cls._sgs is None:
            cls._sgs, cls._name2id = cls._parse_space_groups_xml()
        return cls._sgs, cls._name2id

    @staticmethod
    def _parse_space_groups_xml():

        print("Parsing Space Groups XML")

        ns = {"biojava": "http://www.biojava.org"}

        path = os.path.realpath(__file__)
        directory = os.path.dirname(path)
        subdirectory = os.path.join(directory, "data")
        space_groups_file = os.path.join(subdirectory, "spacegroups.xml")

        tree = ET.parse(space_groups_file)
        root = tree.getroot()
        sgs = {}
        name2id = {}
        for item in root.find("mapProperty"):
            node = item.find("biojava:SpaceGroup", ns)
            key = int(item.find("key").text)
            short_name = node.find("shortSymbol").text
            alt_short_name_node = node.find("altShortSymbol")
            alt_short_name = (
                alt_short_name_node.text if alt_short_name_node is not None else None
            )
            sg = SpaceGroup(
                key,
                int(node.find("multiplicity").text),
                int(node.find("primitiveMultiplicity").text),
                short_name,
                alt_short_name,
                [x.text for x in node.find("transfAlgebraic").findall("item")],
                BravaisLattice[node.find("bravLattice").text],
            )
            sgs[key] = sg
            name2id[short_name] = key
            if alt_short_name is not None:
                name2id[alt_short_name] = key

        return sgs, name2id

    @classmethod
    def get_space_group_by_id(cls, id):
        """Get the space group for the given standard identifier.

        See for example http://en.wikipedia.org/wiki/Space_group
        """
        sgs, _ = cls.get_space_groups()
        return sgs[id]

    @classmethod
    def get_space_group_by_name(cls, short_name):
        """Get the space group for the given international short short_name.

        Use the PDB format, e.g. 'P 21 21 21' or 'C 1 c 1'
        """
        if (short_name is None) or (len(short_name) <= 2):
            return None
        if short_name == "P 1-":
            short_name = "P -1"
        short_name = short_name[0] + short_name[1:].lower()
        sgs, name2id = cls.get_space_groups()
        return sgs[name2id[short_name]]


class SpaceGroup:
    """Class to handle space groups.

    A crystallographic space group. We store the standard numeric identifier,
    the international short symbol and the transformations corresponding to
    each space group (as Matrix4ds and in algebraic notation).
    The information for all (protein crystallography) space groups can be
    parsed from the XML file in the data directory.

    See: http://en.wikipedia.org/wiki/Space_group
    """

    _split_pat1 = re.compile("((?:[+-]?[XYZ])+)([+-][0-9/.]+)")
    _split_pat2 = re.compile("([+-]?[0-9/.]+)((?:[+-][XYZ])+)")
    _coord_pat = re.compile("(?:([+-])?([XYZ]))+?")
    _transf_coef_pat = re.compile("([-+]?[0-9.]+)(?:/([0-9.]+))?")
    _non_enant_pat = re.compile("[-abcmnd]")

    DELTA = 0.0000001

    def __init__(
        self,
        id,
        multiplicity,
        primitive_multiplicity,
        short_symbol,
        alt_short_symbol,
        transf_algebraic,
        bravais_lattice,
    ):
        """Initialize a SpaceGroup object."""
        self.id = id
        self.multiplicity = multiplicity
        self.primitive_multiplicity = primitive_multiplicity
        self.short_symbol = short_symbol
        self.alt_short_symbol = alt_short_symbol
        self.transf_algebraic = transf_algebraic
        self.bravais_lattice = bravais_lattice

        self._transformations = None
        self._cell_translations = None  # in space groups I, C, F or H there are pure cell translations corresponding to recenterings

        self._axis_angles = None
        self._axis_types = None  # indices of array are transformIds

    def is_enantiomorphic(self):
        """Test if this space group is enantiomorphic."""
        m = self._non_enant_pat.search(self.short_symbol)
        if m is not None:
            return False
        else:
            return True

    def get_transformations(self):
        """Get transformation matrices."""
        if self._transformations is not None:
            return self._transformations

        self._transformations = []
        for transf in self.transf_algebraic:
            self._transformations.append(self._get_matrix_from_algebraic(transf))

        return self._transformations

    @staticmethod
    def parse_space_group(short_name):
        """Get the space group for the given international short name.

        Use the PDB format, e.g. 'P 21 21 21' or 'C 1 c 1'
        """
        return SymoplibParser.get_space_group_by_name(short_name)

    def get_cell_translations(self):
        """Return cell translations."""
        if (self._cell_translations is not None) and (len(self._cell_translations) > 0):
            return self._cell_translations

        if self.multiplicity == self.primitive_multiplicity:
            return

        fold = int(self.multiplicity / self.primitive_multiplicity)
        self._cell_translations = [None] * fold
        self._cell_translations[0] = np.array([0, 0, 0])
        transformations = self.get_transformations()
        for n in range(1, fold):
            if len(transformations) < n * self.primitive_multiplicity:
                warnings.warn(
                    f"WARNING number of transformations < {n * self.primitive_multiplicity}",
                    BiopythonWarning,
                )
                t = transformations[n * self.primitive_multiplicity]
                self._cell_translations[n] = t[:3, 3]

        return self._cell_translations

    def get_brav_lattice(self):
        """Return Bravais lattice."""
        return self.bravais_lattice

    def get_cell_translation(self, n):
        """Return cell translation."""
        return self.get_cell_translations()[n]

    def get_multiplicity(self):
        """Return multiplicity."""
        return self.multiplicity

    def get_primitive_multiplicity(self):
        """Return primitive multiplicity."""
        return self.primitive_multiplicity

    def get_id(self):
        """Return id.

        Gets the standard numeric identifier for the space group.
            See for example http://en.wikipedia.org/wiki/Space_group
            or the IUCr crystallographic tables
        """
        return self.id

    def get_short_symbol(self):
        """Return short symbol.

        Gets the international short name (as used in PDB),
            e.g. "P 21 21 21" or "C 1 c 1"
        """
        return self.short_symbol

    def get_alt_short_symbol(self):
        """Return alt short symbol.

        Gets the alternative international short name (as sometimes used in PDB),
            e.g. "I 1 2 1" instead of "I 2"
        """
        return self.alt_short_symbol

    def _get_rot_axes_and_angles(self):
        if self._axis_angles is not None:
            return self._axis_angles
        self._axis_angles = [None] * self.multiplicity
        self._axis_angles[0] = (0, Vector(0, 0, 0))
        transformations = self.get_transformations()
        for i in range(1, len(transformations)):
            r = transformations[i][:3, :3]
            self._axis_angles[i] = self._get_rot_axis_and_angle(r)  # m2rotaxis(r)
        return self._axis_angles

    def _get_axis_fold_types(self):
        """Calculate axis fold type for rotations.

        Calculates the axis fold type (1, 2, 3, 4, 5, 6 for rotations or -1, -2, -3, -4, -6 improper rotations)
            from the trace of the rotation matrix, see for instance
            http://www.crystallography.fr/mathcryst/pdf/Gargnano/Aroyo_Gargnano_1.pdf
        """
        if self._axis_types is not None:
            return self._axis_types
        self._axis_types = [None] * self.multiplicity
        transformations = self.get_transformations()
        for i in range(len(transformations)):
            self._axis_types[i] = self.get_rot_axis_type(transformations[i])
        return self._axis_types

    def get_axis_fold_type(self, transform_id):
        """Return axis fold for transform.

        Given a transformId returns the type of axis of rotation: 1 (no rotation), 2, 3, 4 or 6 -fold
            and for improper rotations: -1, -2, -3, -4 and -6
        """
        axis_folds = self._get_axis_fold_types()
        return axis_folds[transform_id]

    def get_rot_axis_angle(self, transform_id):
        """Get rotation axis."""
        axis_angles = self._get_rot_axes_and_angles()
        return axis_angles[transform_id]

    def are_in_same_axis(self, t_id_1, t_id_2):
        """Return true if both given transform ids belong to the same crystallographic axis (a, b or c).

        For two non-rotation transformations (i.e. identity operators) returns true.
        """
        if t_id_1 == t_id_2:
            return True

        axis_angles = self._get_rot_axes_and_angles()
        axis_folds = self._get_axis_fold_types()

        if (axis_folds[t_id_1] == 1) and (axis_folds[t_id_2] == 1):
            return True

        # we can't deal yet with improper rotations: we return false whenever either of them is improper
        if axis_folds[t_id_1] < 0 or axis_folds[t_id_2] < 0:
            return False

        axis1 = axis_angles[t_id_1][1]
        axis2 = axis_angles[t_id_2][1]

        # TODO revise: we might need to consider that the 2 are in same direction but opposite senses
        #  the method is not used at the moment anyway
        if np.abs(axis1.angle(axis2)) < self.DELTA:
            return True

        return False

    def get_transformation(self, id):
        """Return transformation.

        Gets a transformation by index expressed in crystal axes basis.
            Index 0 corresponds always to the identity transformation.
            Beware the returned Matrix4d is not a copy but it stays linked
            to the one stored in this SpaceGroup object
        """
        transformations = self.get_transformations()
        return transformations[id]

    def get_transf_algebraic(self, id):
        """Get a transformation algebraic string given its index.

        Index 0 corresponds always to the identity transformation.
        """
        return self.transf_algebraic[id]

    def __hash__(self):
        return 31 + self.id  # ??? from the java...

    def __equals__(self, other):
        if not isinstance(other, SpaceGroup):
            return False
        return self.id == other.id

    def get_num_operators(self):
        """Get the number of symmetry operators corresponding to this SpaceGroup (counting the identity operator)."""
        return len(self.get_transformations())

    @classmethod
    def get_algebraic_from_matrix(cls, m):
        """Return algebraic representation of transformation matrix."""
        x = cls._format_alg(m[0, :])
        y = cls._format_alg(m[1, :])
        z = cls._format_alg(m[2, :])
        return f"{x},{y},{z}"

    @classmethod
    def _format_alg(cls, v):
        leading = v[:3] != 0
        x = (
            ""
            if np.abs(v[0]) < cls.DELTA
            else cls._format_coeff(v[0], leading[0]) + "X"
        )
        y = (
            ""
            if np.abs(v[1]) < cls.DELTA
            else cls._format_coeff(v[1], leading[1]) + "Y"
        )
        z = (
            ""
            if np.abs(v[2]) < cls.DELTA
            else cls._format_coeff(v[2], leading[2]) + "Z"
        )
        t = "" if np.abs(v[3]) < cls.DELTA else cls._format_trans(v[3])
        return x + y + z + t

    @classmethod
    def _format_coeff(cls, c, leading):
        if leading:
            return (
                ("" if c > 0 else "-")
                if np.abs(np.abs(c) - 1) < cls.DELTA
                else f"{c:4.2f}"
            )
        else:
            return (
                ("+" if c > 0 else "-")
                if np.abs(np.abs(c) - 1) < cls.DELTA
                else f"{c:+4.2f}"
            )

    @classmethod
    def _format_trans_coef(cls, c):

        frac = fractions.Fraction(c).limit_denominator(6)

        if np.abs(frac.numerator / frac.denominator - c) > cls.DELTA:
            warnings.warn(
                f"Could not convert {c} to fraction.",
                BiopythonWarning,
            )
            return "+0/0"

        if frac.denominator == 1:
            return f"{frac.numerator:+d}"

        return f"{frac.numerator:+d}/{frac.denominator}"

    @classmethod
    def _get_rot_axis_and_angle(cls, m):
        """Return rotation axis and angle from rotation matrix.

        Given a rotation matrix calculates the rotation axis and angle for it.
        The angle is calculated from the trace, the axis from the eigenvalue
        decomposition.
        If given matrix is improper rotation or identity matrix then
        axis (0,0,0) and angle 0 are returned.
        """
        determinant = np.linalg.det(m)

        if not (np.abs(np.abs(determinant) - 1) < cls.DELTA):
            raise RuntimeError("Given matrix is not a rotation matrix")

        axis_and_angle = (0, Vector(0, 0, 0))

        r = m[:3, :3]
        if not np.allclose(np.linalg.det(r), 1.0):
            # improper rotationL return axis 0,0,0 and angle 0
            return axis_and_angle

        w, v = np.linalg.eig(r)

        if np.allclose(w, 1):
            # the rotation is an identity: we return axis 0,0,0 and angle 0
            return axis_and_angle

        index_of_ev_1 = min(np.where(np.abs(w - 1) < SpaceGroup.DELTA))
        axis = Vector(*(v[index_of_ev_1]))
        angle = np.arccos((np.sum(w) - 1) / 2)
        return (angle, axis)

    @classmethod
    def get_rot_axis_type(cls, m):
        """Return type of rotation.

        Given a transformation matrix containing a rotation returns the type of rotation:
            1 for identity, 2 for 2-fold rotation, 3 for 3-fold rotation, 4 for 4-fold rotation,
            6 for 6-fold rotation,
            -1 for inversions, -2 for mirror planes, -3 for 3-fold improper rotation,
            -4 for 4-fold improper rotation and -6 for 6-fold improper rotation
        """
        axis_type = 0
        rot = m[:3, :3]
        det = np.linalg.det(m)
        if (np.abs(det - 1) >= cls.DELTA) and (np.abs(det + 1) >= cls.DELTA):
            raise RuntimeError("Given matrix does not seem to be a rotation matrix.")

        if det > 0:
            trace = np.trace(rot)
            if trace == 3:
                axis_type = 1
            elif trace == -1:
                axis_type = 2
            elif trace == 0:
                axis_type = 3
            elif trace == 1:
                axis_type = 4
            elif trace == 2:
                axis_type = 6
            else:
                raise RuntimeError(
                    "Trace of transform does not correspond to one of the expected types. This is most likely a bug"
                )
        else:
            trace = np.trace(rot)
            if trace == -3:
                axis_type = -1
            elif trace == 1:
                axis_type = -2
            elif trace == 0:
                axis_type = -3
            elif trace == -1:
                axis_type = -4
            elif trace == -2:
                axis_type = -6
            else:
                raise RuntimeError(
                    "Trace of transform does not correspond to one of the expected types. This is most likely a bug"
                )

        return axis_type

    def __str__(self):
        return self.short_symbol

    @classmethod
    def _get_matrix_from_algebraic(cls, transf_algebraic):
        parts = transf_algebraic.upper().split(",")
        x_coef = cls._convert_algebraic_str_to_coefficients(parts[0].strip())
        y_coef = cls._convert_algebraic_str_to_coefficients(parts[1].strip())
        z_coef = cls._convert_algebraic_str_to_coefficients(parts[2].strip())

        mat = np.eye(4)
        mat[:3, :3] = np.vstack([x_coef[:3], y_coef[:3], z_coef[:3]])
        mat[:3, 3] = [x_coef[3], y_coef[3], z_coef[3]]
        return mat

    @classmethod
    def _convert_algebraic_str_to_coefficients(cls, alg_string):
        letters = None
        noLetters = None
        m = cls._split_pat1.match(alg_string)
        if m is not None:
            letters = m.group(1)
            noLetters = m.group(2)
        else:
            m = cls._split_pat2.match(alg_string)
            if m is not None:
                letters = m.group(2)
                noLetters = m.group(1)
            else:
                letters = alg_string

        coefficients = np.zeros(4)
        matches = cls._coord_pat.findall(letters)
        for m in matches:
            s = -1 if m[0] == "-" else 1
            coord = m[1]
            if coord == "X":
                coefficients[0] = s
            elif coord == "Y":
                coefficients[1] = s
            elif coord == "Z":
                coefficients[2] = s

        if noLetters is not None:
            m = cls._transf_coef_pat.match(noLetters)
            if m is not None:
                num = float(m.group(1))
                den = 1.0
                if m.group(2) is not None:
                    den = float(m.group(2))
                coefficients[3] = num / den
            else:
                coefficients[3] = 0

        return coefficients
