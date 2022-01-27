# TODO: Copyright
#
# This file is part of the Biopython distribution and governed by your
# choice of the "Biopython License Agreement" or the "BSD 3-Clause License".
# Please see the LICENSE file that should have been included as part of this
# package.

"""CrystalCell class, used in xtal module."""

import numpy as np


class CrystalCell:
    """TODO: provide info."""

    MIN_VALID_CELL_SIZE = 10

    def __init__(self, a, b, c, alpha, beta, gamma):
        """Initialize a CrystalCell object.

        :param a: length a
        :type a: float

        :param b: length b
        :type b: float

        :param c: length c
        :type c: float

        :param alpha: angle alpha in deg
        :type alpha: float

        :param beta: angle alpha in deg
        :type beta: float

        :param gamma: angle gamma in deg
        :type gamma: float
        """
        self.a = a
        self.b = b
        self.c = c

        self.set_alpha(alpha)
        self.set_beta(beta)
        self.set_gamma(gamma)

        self._volume = None
        self._max_dimension = 0

        # cached basis change transformation matrices
        self._m = None
        self._m_inv = None
        self._m_transp = None
        self._m_transp_inv = None

    def __eq__(self, other):
        return (
            self.a == other.a
            and self.b == other.b
            and self.c == other.c
            and self.alpha == other.alpha
            and self.beta == other.beta
            and self.gamma == other.gamma
        )

    def __str__(self):
        return f"{self.a:7.2f} {self.b:7.2f} {self.c:7.2f} {self.alpha:6.2f} {self.beta:6.2f} {self.gamma:6.2f}"

    def set_alpha(self, alpha):
        """Set alpha angle.

        :param alpha: alpha angle in degrees
        """
        self.alpha = alpha
        self.alpha_rad = np.deg2rad(alpha)

    def set_beta(self, beta):
        """Set beta angle.

        :param beta: beta angle in degrees
        """
        self.beta = beta
        self.beta_rad = np.deg2rad(beta)

    def set_gamma(self, gamma):
        """Set gamma angle.

        :param gamma: gamma angle in degrees
        """
        self.gamma = gamma
        self.gamma_rad = np.deg2rad(gamma)

    def get_volume(self):
        """Get volume of this unit cell.

        :return: Volume of this unit cell
        :rtype: float
        """
        if self._volume is not None:
            return self._volume

        self._volume = (
            self.a
            * self.b
            * self.c
            * (
                1
                - np.cos(self.alpha_rad) * np.cos(self.alpha_rad)
                - np.cos(self.beta_rad) * np.cos(self.beta_rad)
                - np.cos(self.gamma_rad) * np.cos(self.gamma_rad)
                + 2.0
                * np.cos(self.alpha_rad)
                * np.cos(self.beta_rad)
                * np.cos(self.gamma_rad)
            )
            ** 0.5
        )

        return self._volume

    def get_cell_indices(self, pt):
        """Get the index of a unit cell to which the query point belongs.

        For instance, all points in the unit cell at the origin will return (0,0,0);
        Points in the unit cell one unit further along the `a` axis will return (1,0,0),
        etc.

        :param pt: Input point (in orthonormal coordinates)
        :type pt: ndarray(dtype=float, shape=(3,))

        :return: A new point with the three indices of the cell containing pt
        :rtype: ndarray(dtype=int, shape=(3,))
        """
        return np.floor(self.transf_vec_to_crystal(pt)).astype(int)

    def transf_to_origin_cell(self, pt):
        """Convert the coordinates in pt so that they occur within the (0,0,0) unit cell.

        :type pt: ndarray(dtype=float, shape=(3,))
        :rtype: ndarray(dtype=float, shape=(3,))
        """
        assert pt.shape == (3,)

        p = self.transf_vec_to_crystal(pt)

        lz = p < 0
        p[lz] = (p[lz] % 1.0 + 1.0) % 1.0
        p[~lz] = p[~lz] % 1.0

        return self.transf_vec_to_orthonormal(p)

    def transf_ref_to_origin_cell(self, points, reference):
        """Convert a set of points so that the reference point falls in the unit cell.

        This is useful to transform a whole chain at once, allowing some of the
        atoms to be outside the unit cell, but forcing the centroid to be within it.

        :param points: A set of points to transform (in orthonormal coordinates)
        :type points: list[ndarray(dtype=float, shape=(3,))]

        :param reference: The reference point, which is unmodified but which would
           be in the unit cell were it to have been transformed. It is safe to
           use a member of the points array here.
        :type reference: ndarray(dtype=float, shape=(3,))
        """
        reference = np.floor(self.transf_vec_to_crystal(reference)).astype(int)
        out = []
        for pt in points:
            p = self.transf_vec_to_crystal(pt)
            p -= reference
            out.append(self.transf_vec_to_orthonormal(p))

        return out

    def transf_ref_to_origin_cell_orthonormal(self, ops, reference):
        """Convert a set of points so that the reference point falls in the unit cell.

        :param ops: Set of operations in orthonormal coordinates
        :type ops: list[ndarray(dtype=float, shape=(4,4))]

        :param reference: Reference point, which should be in the unit cell after each operation (also in orthonormal coordinates)
        :type reference: ndarray(dtype=float, shape=(3,))

        :return: A set of orthonormal operators with equivalent rotation to the inputs, but with translation such that the reference point would fall within the unit cell
        :rtype: list[ndarray(dtype=float, shape=(4,4))]
        """
        refXtal = self.transf_vec_to_crystal(reference)
        opsXtal = [None] * len(ops)
        for i in range(len(ops)):
            opsXtal[i] = self.transf_mat_to_crystal(ops[i])

        transformed = self.transf_ref_to_origin_cell_crystal(opsXtal, refXtal)

        for i in range(len(ops)):
            transformed[i] = self.transf_mat_to_orthonormal(transformed[i])

        return transformed

    def transf_ref_to_origin_cell_crystal(self, ops, reference):
        """Convert a set of points so that the reference point falls in the unit cell.

        :param ops: Set of operations in crystal coordinates
        :type ops: list[ndarray(dtype=float, shape=(4,4))]

        :param reference: Reference point, which should be in the unit cell after
         each operation (also in crystal coordinates)
        :type reference: ndarray(dtype=float, shape=(3,))

        :return: A set of crystal operators with equivalent rotation to the
         inputs, but with translation such that the reference point would fall
         within the unit cell
        :rtype: list[ndarray(dtype=float, shape=(4,4))]
        """
        transformed = [None] * len(ops)
        for j in range(len(ops)):
            op = ops[j]
            xXtal = np.floor(op[:3, :3] @ reference + op[:3, 3]).astype(int)

            translation = np.eye(4)
            translation[:3, 3] = -xXtal

            translation = translation @ op

            # This code has no effect (?)
            # Point3d ref2 = new Point3d(reference);
            # translation.transform(ref2);

            transformed[j] = translation

        return transformed

    def transf_vec_to_orthonormal(self, v):
        """Transform the given crystal basis coordinates into orthonormal coordinates.

        e.g. transfToOrthonormal(new Point3d(1,1,1)) returns the orthonormal coordinates of the
        vertex of the unit cell.
        See Giacovazzo section 2.E, eq. 2.E.1 (or any linear algebra manual)

        :param v: 3 vector in crystal coordinates
        :type v: ndarray(dtype=float, shape=(3,))

        :return: 3 vector in orthonormal coordinates
        :rtype: ndarray(dtype=float, shape=(3,))
        """
        return self._get_m_transpose_inv() @ v

    def transf_mat_to_orthonormal(self, m):
        """Transform given Matrix4d in crystal basis to the orthonormal basis using the PDB axes convention (NCODE=1).

        :param m: 4x4 matrix in crystal coordinates
        :type m: ndarray(dtype=float, shape=(4,4))

        :return: 4x4 matrix in orthonormal coordinates
        :rtype: ndarray(dtype=float, shape=(4,4))
        """
        trans = m[:3, 3]
        trans = self.transf_vec_to_orthonormal(trans)
        rot = self._get_m_transpose_inv() @ m[:3, :3] @ self.get_m_transpose()
        out = np.zeros((4, 4))
        out[:3, :3] = rot
        out[:3, 3] = trans
        out[3, 3] = 1
        return out

    def transf_vec_to_crystal(self, v):
        """Transform the given crystal basis coordinates into orthonormal coordinates.

        e.g. transfToOrthonormal(new Point3d(1,1,1)) returns the orthonormal coordinates of the
        vertex of the unit cell.

        See Giacovazzo section 2.E, eq. 2.E.1 (or any linear algebra manual)

        :param v: 3 vector in orthonormal coordinates
        :type v: ndarray(dtype=float, shape=(3,))

        :return: 3 vector in crystal coordinates
        :rtype: ndarray(dtype=float, shape=(3,))
        """
        return self.get_m_transpose() @ v

    def transf_mat_to_crystal(self, m):
        """Transform given 4x4 matrix in orthonormal basis to the crystal basis using the PDB axes convention (NCODE=1).

        :param m: 4x4 matrix in orthonormal coordinates
        :type m: ndarray(dtype=float, shape=(4,4))

        :return: 4x4 matrix in crystal coordinates
        :rtype: ndarray(dtype=float, shape=(4,4))
        """
        trans = m[:3, 3]
        trans = self.transf_vec_to_crystal(trans)
        rot = self.get_m_transpose() @ m[:3, :3] @ self._get_m_transpose_inv()
        out = np.zeros((4, 4))
        out[:3, :3] = rot
        out[:3, 3] = trans
        out[3, 3] = 1
        return out

    def _get_m_inv(self):
        """Return the change of basis (crystal to orthonormal) transform matrix.

        Returns M inverse in the notation of Giacovazzo.
        Using the PDB axes convention (CCP4 uses NCODE identifiers to distinguish the different conventions, the PDB one is called NCODE=1)
        The matrix is only calculated upon first call of this method, thereafter it is cached.
        See "Fundamentals of Crystallography" C. Giacovazzo, section 2.5 (eq 2.30)

        The non-standard orthogonalisation codes (NCODE for ccp4) are flagged in REMARK 285 after 2011's remediation
        with text: "THE ENTRY COORDINATES ARE NOT PRESENTED IN THE STANDARD CRYSTAL FRAME". There were only 148 PDB
        entries with non-standard code in 2011. See:
        http://www.wwpdb.org/documentation/2011remediation_overview-061711.pdf
        The SCALE1,2,3 records contain the correct transformation matrix (what Giacovazzo calls M matrix).
        In those cases if we calculate the M matrix following Giacovazzo's equations here, we get an entirely wrong one.
        Examples of PDB with non-standard orthogonalisation are 1bab and 1bbb.

        :return: Minv
        :rtype: ndarray(dtype=float, shape=(3,3))
        """
        if self._m_inv is not None:
            return self._m_inv

        self._m_inv = np.array(
            [
                [self.a, 0, 0],
                [self.b * np.cos(self.gamma_rad), self.b * np.sin(self.gamma_rad), 0],
                [
                    self.c * np.cos(self.beta_rad),
                    -self.c * np.sin(self.beta_rad) * self._get_cos_alpha_star(),
                    1.0 / self._get_c_star(),
                ],
            ]
        )

        return self._m_inv

    def _get_c_star(self):
        return (self.a * self.b * np.sin(self.gamma_rad)) / self.get_volume()

    def _get_cos_alpha_star(self):
        return (
            np.cos(self.beta_rad) * np.cos(self.gamma_rad) - np.cos(self.alpha_rad)
        ) / (np.sin(self.beta_rad) * np.sin(self.gamma_rad))

    def _get_m(self):
        if self._m is not None:
            return self._m
        self._m = np.linalg.inv(self._get_m_inv())
        return self._m

    def get_m_transpose(self):
        """Return m transpose."""
        if self._m_transp is not None:
            return self._m_transp
        m = self._get_m()
        self._m_transp = m.T
        return self._m_transp

    def _get_m_transpose_inv(self):
        if self._m_transp_inv is not None:
            return self._m_transp_inv
        self._m_transp_inv = np.linalg.inv(self.get_m_transpose())
        return self._m_transp_inv

    def _distance(a, b):
        return sum((a - b) ** 2) ** 0.5

    def get_max_dimension(self):
        """Return max dimension of cell."""
        if self._max_dimension != 0:
            return self._max_dimension
        import itertools

        vecs = [np.array(x) for x in map(list, itertools.product([0, 1], repeat=3))]
        verts = [self.transf_vec_to_orthonormal(x) for x in vecs]
        assert np.allclose(verts[0], 0)
        vert_dists = [self._distance(verts[i], verts[7 - i]) for i in range(4)]
        return max(vert_dists)

    def check_scale_matrix_consistency(self, scale_matrix):
        """Check scale matrix consistency.

        Given a scale matrix parsed from the PDB entry (SCALE1,2,3 records),
        checks that the matrix is a consistent scale matrix by comparing the
        cell volume to the inverse of the scale matrix determinant (tolerance of 1/100).
        If they don't match false is returned.
        See the PDB documentation for the SCALE record.
        See also last equation of section 2.5 of "Fundamentals of Crystallography" C. Giacovazzo

        :type scaleMatrix: ndarray(dtype=float, shape=(4,4))
        :rtype: bool
        """
        vol = self.get_volume()
        m = scale_matrix[:3, :3]

        # note we need to have a relaxed tolerance here as the PDB scale matrix is given with not such high precision
        # plus we don't want to have false positives, so we stay conservative
        tolerance = vol / 100.0
        if np.abs(vol - 1.0 / np.linalg.det(m)) > tolerance:
            return False

        # this would be to check our own matrix, must always match!
        # if (!deltaComp(vol,1.0/getMTranspose().determinant())) {
        # 	System.err.println("Our calculated SCALE matrix does not match 1/det=cell volume");
        # }

        return True

    def check_scale_matrix(self, scale_matrix):
        """Check scale matrix consistency.

        Given a scale matrix parsed from a PDB entry (SCALE1,2,3 records),
        compares it to our calculated Mtranspose matrix to see if they coincide and
        returns true if they do.
        If they don't that means that the PDB entry is not in the standard
        orthogonalisation (NCODE=1 in ccp4).
        In 2011's remediation only 148 PDB entries were found not to be in
        a non-standard orthogonalisation. See:
        http://www.wwpdb.org/documentation/2011remediation_overview-061711.pdf
        For normal cases the scale matrix is diagonal without a translation component.
        Additionally the translation component of the SCALE matrix is also checked to
        make sure it is (0,0,0), if not false is return

        :type scaleMatrix: ndarray(dtype=float, shape=(4,4))
        :rtype: bool
        """
        m_transpose = self.get_m_transpose()

        if not np.allclose(m_transpose[:3, :3], scale_matrix[:3, :3]):
            return False

        if not np.allclose(scale_matrix[:3, 3], 0):
            return False

        return True

    def is_cell_reasonable(self):
        """Check if cell dimensions are reasonable.

        Checks whether the dimensions of this crystal cell are reasonable for protein
        crystallography: if all 3 dimensions are below {@value #MIN_VALID_CELL_SIZE} the cell
        is considered unrealistic and false returned

        :rtype: bool
        """
        if (
            (self.a < self._mIN_VALID_CELL_SIZE)
            and (self.b < self._mIN_VALID_CELL_SIZE)
            and (self.c < self._mIN_VALID_CELL_SIZE)
        ):
            return False
        return True
