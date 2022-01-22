from re import M
import numpy as np

class CrystalCell:

    MIN_VALID_CELL_SIZE = 10

    def __init__(
        self,
        a,
        b,
        c,
        alpha,
        beta,
        gamma):

        self.a = a
        self.b = b
        self.c = c
        
        self.setAlpha(alpha)
        self.setBeta(beta)
        self.setGamma(gamma)

        self.volume = None
        self.maxDimension = 0
        
        # cached basis change transformation matrices
        self.M = None
        self.Minv = None
        self.Mtransp = None
        self.MtranspInv = None

    def setAlpha(self, alpha):
        """Set alpha angle
        
        :param alpha: alpha angle in degrees
        """
        self.alpha = alpha
        self.alphaRad = np.deg2rad(alpha)

    def setBeta(self, beta):
        """Set beta angle
        
        :param beta: beta angle in degrees
        """
        self.beta = beta
        self.betaRad = np.deg2rad(beta)

    def setGamma(self, gamma):
        """Set gamma angle
        
        :param gamma: gamma angle in degrees
        """
        self.gamma = gamma
        self.gammaRad = np.deg2rad(gamma)

    def getVolume(self):
        """Get volume of this unit cell
        
        :return: Volume of this unit cell
        :rtype: float
        """
        if (self.volume != None): return self.volume
        
        self.volume = self.a * self.b * self.c \
                       * (1 - np.cos(self.alphaRad)*np.cos(self.alphaRad) \
                            - np.cos(self.betaRad) * np.cos(self.betaRad) \
                            - np.cos(self.gammaRad) * np.cos(self.gammaRad) \
                            + 2.0 * np.cos(self.alphaRad) * np.cos(self.betaRad) * np.cos(self.gammaRad)
                         ) ** 0.5
        
        return self.volume

    def getCellIndices(self, pt):
        """Get the index of a unit cell to which the query point belongs.
        
        For instance, all points in the unit cell at the origin will return (0,0,0);
        Points in the unit cell one unit further along the `a` axis will return (1,0,0),
        etc.
        
        :param pt: Input point (in orthonormal coordinates)
        :type pt: ndarray(dtype=float, shape=(3,))

        :return: A new point with the three indices of the cell containing pt
        :rtype: ndarray(dtype=int, shape=(3,))
        """

        return self.transfVecToCrystal(pt).astype(int)

    def transfToOriginCell(self, pt):
        """Converts the coordinates in pt so that they occur within the (0,0,0) unit cell
        
        :type pt: ndarray(dtype=int, shape=(3,))
        :rtype: ndarray(dtype=int, shape=(3,))
        """
        assert pt.shape == (3,)

        p = self.transfVecToCrystal(pt)

        lz = p < 0
        p[lz] = (p[lz] % 1.0 + 1.0) % 1.0
        p[~lz] = p[~lz] % 1.0

        return self.transfVecToOrthonormal(p)

    def transfRefToOriginCell(self, points, reference):
        """Converts a set of points so that the reference point falls in the unit cell.
        
        This is useful to transform a whole chain at once, allowing some of the
        atoms to be outside the unit cell, but forcing the centroid to be within it.
        
        :param points: A set of points to transform (in orthonormal coordinates)
        :type points: list[ndarray(dtype=float, shape=(3,))]

        :param reference: The reference point, which is unmodified but which would
           be in the unit cell were it to have been transformed. It is safe to
           use a member of the points array here.
        :type reference: ndarray(dtype=float, shape=(3,))
        """

        reference = self.transfVecToCrystal(reference).astype(int)

        for pt in points:
            p = self.transfVecToCrystal(pt)
            p = -p
            pt[:] = self.transfVecToOrthonormal(p)

    def transfRefToOriginCellOrthonormal(self, ops, reference):
        """
        :param ops: Set of operations in orthonormal coordinates
        :type ops: list[ndarray(dtype=float, shape=(4,4))]

        :param reference: Reference point, which should be in the unit cell after
        each operation (also in orthonormal coordinates)
        :type reference: ndarray(dtype=float, shape=(3,))

        :return: A set of orthonormal operators with equivalent rotation to the
        inputs, but with translation such that the reference point would fall
        within the unit cell
        :rtype: list[ndarray(dtype=float, shape=(4,4))]
        """

        refXtal = self.transfVecToCrystal(reference)
        opsXtal = [None] * ops.length
        for i in range(ops.length):
            opsXtal[i] = self.transfMatToCrystal(ops[i])
        
        transformed = self.transfRefToOriginCellCrystal(opsXtal, refXtal)

        for i in range(ops.length):
            transformed[i] = self.transfMatToOrthonormal(transformed[i])

        return transformed
    
    def transfRefToOriginCellCrystal(self, ops, reference):
        """
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

        transformed = [None] * ops.length
        for j in range(ops.length):
            op = ops[j]
            xXtal = (op @ reference).astype(int)
            
            translation = np.eye(4)
            translation[:3, 3] = -xXtal

            translation @= op

            # This code has no effect (?)
            # Point3d ref2 = new Point3d(reference);
			# translation.transform(ref2);

            transformed[j] = translation

        return transformed

    def transfVecToOrthonormal(self, v):
        """Transforms the given crystal basis coordinates into orthonormal coordinates.
	    e.g. transfToOrthonormal(new Point3d(1,1,1)) returns the orthonormal coordinates of the
	    vertex of the unit cell.
	    See Giacovazzo section 2.E, eq. 2.E.1 (or any linear algebra manual)
        
        :param v: 3 vector in crystal coordinates
        :type v: ndarray(dtype=float, shape=(3,))

        :return: 3 vector in orthonormal coordinates
        :rtype: ndarray(dtype=float, shape=(3,))
        """
        return self.getMTransposeInv() @ v

    def transfMatToOrthonormal(self, m):
        """Transform given Matrix4d in crystal basis to the orthonormal basis using
	    the PDB axes convention (NCODE=1)
        
        :param m: 4x4 matrix in crystal coordinates 
        :type m: ndarray(dtype=float, shape=(4,4))

        :return: 4x4 matrix in orthonormal coordinates
        :rtype: ndarray(dtype=float, shape=(4,4))
        """
        trans = m[:3, 3]
        trans = self.transfVecToOrthonormal(trans)
        rot =  self.getMTransposeInv() @ m[:3, :3] @ self.getMTranspose()
        out = np.zeros((4, 4))
        out[:3, :3] = rot
        out[:3, 3] = trans
        out[3, 3] = 1
        return out

    def transfVecToCrystal(self, v):
        """ Transforms the given crystal basis coordinates into orthonormal coordinates.
	    e.g. transfToOrthonormal(new Point3d(1,1,1)) returns the orthonormal coordinates of the
	    vertex of the unit cell.
	    
        See Giacovazzo section 2.E, eq. 2.E.1 (or any linear algebra manual)
        
        :param v: 3 vector in orthonormal coordinates
        :type v: ndarray(dtype=float, shape=(3,))

        :return: 3 vector in crystal coordinates
        :rtype: ndarray(dtype=float, shape=(3,))
        """
        return self.getMTranspose() @ v

    def transfMatToCrystal(self, m):
        """Transform given 4x4 matrix in orthonormal basis to the crystal basis using
	    the PDB axes convention (NCODE=1)
        
        :param m: 4x4 matrix in orthonormal coordinates 
        :type m: ndarray(dtype=float, shape=(4,4))

        :return: 4x4 matrix in crystal coordinates
        :rtype: ndarray(dtype=float, shape=(4,4))
        """
        trans = m[:3, 3]
        trans = self.transfVecToCrystal(trans)
        rot =  self.getMTranspose() @ m[:3, :3] @ self.getMTransposeInv()
        out = np.zeros((4, 4))
        out[:3, :3] = rot
        out[:3, 3] = trans
        out[3, 3] = 1
        return out

    def getMInv(self):
        """Returns the change of basis (crystal to orthonormal) transform matrix, that is
        M inverse in the notation of Giacovazzo.
        Using the PDB axes convention
        (CCP4 uses NCODE identifiers to distinguish the different conventions, the PDB one is called NCODE=1)
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
        if self.Minv is not None:
            return self.Minv
        
        self.Minv = np.array([
            [self.a, 0, 0],
            [self.b * np.cos(self.gammaRad), self.b * np.sin(self.gammaRad), 0],
            [self.c * np.cos(self.betaRad), -self.c * np.sin(self.betaRad) * self.getCosAlphaStar(), 1.0 / self.getCstar()]
        ])

        return self.Minv

    def getCstar(self):
        return (self.a*self.b*np.sin(self.gammaRad))/self.getVolume()

    def getCosAlphaStar(self):
        return (np.cos(self.betaRad)*np.cos(self.gammaRad)-np.cos(self.alphaRad))/(np.sin(self.betaRad)*np.sin(self.gammaRad))

    def getM(self):
        if self.M is not None: return self.M
        self.M = np.inv(self.getMInv())
        return self.M

    def getMTranspose(self):
        if self.Mtransp is not None: return self.Mtransp
        M = self.getM()
        self.Mtransp = M.T
        return self.Mtransp

    def getMTransposeInv(self):
        if self.MtranspInv is not None: return self.MtranspInv
        self.MtranspInv = np.inv(self.getMTranspose())
        return self.MtranspInv

    def getMaxDimension(self):
        if self.maxDimension != 0: return self.maxDimension
        import itertools
        vecs  = [np.array(x) for x in map(list, itertools.product([0, 1], repeat=3))]
        verts = [self.transfVecToOrthonormal(x) for x in vecs]
        assert np.allclose(verts[0], 0)
        distance = lambda a, b: sum((a-b)**2)**0.5
        vertDists = [distance(verts[i], verts[7-i]) for i in range(4)]
        return max(vertDists)

    def checkScaleMatrixConsistency(self, scaleMatrix):
        """Given a scale matrix parsed from the PDB entry (SCALE1,2,3 records),
        checks that the matrix is a consistent scale matrix by comparing the
        cell volume to the inverse of the scale matrix determinant (tolerance of 1/100).
        If they don't match false is returned.
        See the PDB documentation for the SCALE record.
        See also last equation of section 2.5 of "Fundamentals of Crystallography" C. Giacovazzo
        
        :type scaleMatrix: ndarray(dtype=float, shape=(4,4))
        :rtype: bool

        """
        
        vol = self.getVolume()
        m = scaleMatrix[:3, :3]

        # note we need to have a relaxed tolerance here as the PDB scale matrix is given with not such high precision
        # plus we don't want to have false positives, so we stay conservative
        tolerance = vol/100.0
        if np.abs(vol - 1.0/np.det(m)) > tolerance:
            return False
        
        # this would be to check our own matrix, must always match!
        # if (!deltaComp(vol,1.0/getMTranspose().determinant())) {
        # 	System.err.println("Our calculated SCALE matrix does not match 1/det=cell volume");
        # }

        return True        


	 

    def checkScaleMatrix(self, scaleMatrix):
        """Given a scale matrix parsed from a PDB entry (SCALE1,2,3 records),
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

        mTranspose = self.getMTranspose()
        
        if not np.allclose(mTranspose[:3, :3], scaleMatrix[:3, :3]):
            return False

        if not np.allclose(scaleMatrix[:3, 3], 0):
            return False

        return True

# TODO: 

# /**
# 	 * Checks whether the dimensions of this crystal cell are reasonable for protein
# 	 * crystallography: if all 3 dimensions are below {@value #MIN_VALID_CELL_SIZE} the cell
# 	 * is considered unrealistic and false returned
# 	 * @return
# 	 */
# 	public boolean isCellReasonable() {
# 		// this check is necessary mostly when reading PDB files that can contain the default 1 1 1 crystal cell
# 		// if we use that further things can go wrong, for instance for interface calculation
# 		// For instance programs like coot produce by default a 1 1 1 cell

# 		if (this.getA()<MIN_VALID_CELL_SIZE &&
# 				this.getB()<MIN_VALID_CELL_SIZE &&
# 				this.getC()<MIN_VALID_CELL_SIZE) {
# 			return false;
# 		}

# 		return true;

# 	}

# 	@Override
# 	public String toString() {
# 		return String.format(Locale.US, "a%7.2f b%7.2f c%7.2f alpha%6.2f beta%6.2f gamma%6.2f", a, b, c, alpha, beta, gamma);
# 	}