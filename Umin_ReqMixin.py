"""
This file will contain the class U_min_finding
"""


import math
from matplotlib import pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import r2_score, mean_squared_error
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import RBFInterpolator
import constants
from scipy.optimize import minimize, BFGS, basinhopping
from scipy.optimize import check_grad
from itertools import product
# from numba import njit, prange


class Umin_ReqMixin:
    """
    This class will be inherited by the simulation class and will contain all the needed to find Umin.
    This section will also house the fucntions related to the polynomial
    """

    def _ensure_dc_center_fit(self, polyfit_deg: int = 4):
        """
        Make sure we have a center-region polynomial fit for the DC (Static_TotalV).
        Stores it in self.center_fits[self.trapVariables.dc_key].
        """
        dk = self.trapVariables.dc_key
        if dk not in self.center_fits or self.center_fits[dk] is None:
            self.center_fits[dk] = self.get_voltage_poly_for_drive_at_region(
                dk,
                region_x_low=-constants.center_region_x_um,
                region_x_high=constants.center_region_x_um,
                region_y_low=-constants.center_region_y_um,
                region_y_high=constants.center_region_y_um,
                region_z_low=-constants.center_region_z_um,
                region_z_high=constants.center_region_z_um,
                polyfit=polyfit_deg,
            )

    def evaluate_center_poly(self, x, y, z):
        self._ensure_dc_center_fit()
        center_model, center_poly, _ = self.center_fits[self.trapVariables.dc_key]
        X = center_poly.transform(np.array([[x, y, z]]))
        return float(center_model.predict(X)[0])

    def get_center_poly_c_vector(self, polyfit_deg: int = 4, return_powers: bool = False):
        """
        Ensure the DC center-region polynomial fit exists and return its coefficient vector.

        The coefficient order matches sklearn's PolynomialFeatures include_bias=True,
        i.e. the order in center_poly.powers_ (lexicographic by total degree).

        Term order examples (degree=4)::
            1,
            x, y, z,
            x^2, x y, x z, y^2, y z, z^2,
            x^3, x^2 y, x^2 z, x y^2, x y z, x z^2, y^3, y^2 z, y z^2, z^3,
            x^4, x^3 y, x^3 z, x^2 y^2, x^2 y z, x^2 z^2,
            x y^3, x y^2 z, x y z^2, x z^3,
            y^4, y^3 z, y^2 z^2, y z^3, z^4

        So: xy^2 is after x^2 z (it appears in the degree-3 block as x y^2),
        and x^3 y z (degree=5 term) is NOT included for polyfit_deg=4.
        If you need explicit term positions, pass return_powers=True.

        Args:
            polyfit_deg: polynomial degree for the center fit if it needs to be created.
            return_powers: if True, also return the integer powers array (n_terms, 3).

        Returns:
            c (np.ndarray) or (c, powers) if return_powers=True.
        """
        self._ensure_dc_center_fit(polyfit_deg=polyfit_deg)
        center_model, center_poly, _ = self.center_fits[self.trapVariables.dc_key]
        c = np.asarray(center_model.coef_, dtype=float).copy()
        if return_powers:
            return c, np.asarray(center_poly.powers_, dtype=int).copy()
        return c

    def evaluate_center_poly_1stderivatives(self, x, y, z):
        self._ensure_dc_center_fit()
        center_model, center_poly, _ = self.center_fits[self.trapVariables.dc_key]
        powers = center_poly.powers_
        coefs = center_model.coef_
        pt = np.array([x, y, z])
        derivs = np.zeros(3)
        for i in range(3):
            acc = 0.0
            for j, p in enumerate(powers):
                if p[i] == 0:
                    continue
                term = coefs[j] * p[i]
                for k in range(3):
                    exp = p[k] - (1 if (k == i) else 0)
                    if exp != 0:
                        term *= pt[k] ** exp
                acc += term
            derivs[i] = acc
        return tuple(derivs)

    def evaluate_center_poly_2ndderivatives(self, x, y, z):
        """
        Returns full 3x3 symmetric Hessian matrix of second partial derivatives
        at point (x, y, z): H[i,j] = d²V / dxi dxj
        """
        self._ensure_dc_center_fit()

        center_model, center_poly, _ = self.center_fits[self.trapVariables.dc_key]
        powers = center_poly.powers_
        coefs = center_model.coef_
        point = np.array([x, y, z])

        hessian = np.zeros((3, 3))

        for i in range(3):
            for j in range(3):
                d2p = 0.0
                for k, power in enumerate(powers):
                    if power[i] < 1 or power[j] < 1:
                        continue
                    if i == j and power[i] < 2:
                        continue

                    coeff = coefs[k]
                    if i == j:
                        coeff *= power[i] * (power[i] - 1)
                    else:
                        coeff *= power[i] * power[j]

                    term = coeff * np.prod(
                        [
                            point[m]
                            ** (
                                power[m]
                                if m not in [i, j]
                                else power[m]
                                - (1 if m in [i, j] else 0)
                                - (1 if (i == j and m == i) else 0)
                            )
                            for m in range(3)
                        ]
                    )
                    d2p += term
                hessian[i, j] = d2p

        return hessian

    def evaluate_center_poly_3rdderivatives(self, x, y, z):
        """
        Returns full 3x3x3 third derivative tensor: T[i,j,k] = d³V / dxi dxj dxk
        Symmetric in all indices.
        """
        self._ensure_dc_center_fit()

        center_model, center_poly, _ = self.center_fits[self.trapVariables.dc_key]
        powers = center_poly.powers_
        coefs = center_model.coef_
        point = np.array([x, y, z])

        third_tensor = np.zeros((3, 3, 3))

        for i in range(3):
            for j in range(3):
                for k in range(3):
                    d3p = 0.0
                    for l, power in enumerate(powers):
                        if power[i] < 1 or power[j] < 1 or power[k] < 1:
                            continue

                        count_i = power[i] - (1 if i == j else 0) - (1 if i == k else 0)
                        count_j = power[j] - (1 if j == k else 0)
                        count_k = power[k]

                        if i == j == k and power[i] < 3:
                            continue
                        elif (
                            len({i, j, k}) == 2
                            and min(power[i], power[j], power[k]) < 2
                        ):
                            continue
                        elif (
                            len({i, j, k}) == 3
                            and min(power[i], power[j], power[k]) < 1
                        ):
                            continue

                        coeff = coefs[l]
                        coeff *= power[i] * power[j] * power[k]

                        term = coeff * np.prod(
                            [
                                point[m]
                                ** (
                                    power[m]
                                    - (
                                        [i, j, k].count(m)
                                    )  # subtract order for each partial
                                )
                                for m in range(3)
                            ]
                        )
                        d3p += term
                    third_tensor[i, j, k] = d3p

        return third_tensor

    def evaluate_center_poly_4thderivatives(self, x, y, z):
        """
        Returns full 3x3x3x3 fourth derivative tensor: T[i,j,k,l] = d⁴V / dxi dxj dxk dxl
        Symmetric in all indices.
        """
        self._ensure_dc_center_fit()

        center_model, center_poly, _ = self.center_fits[self.trapVariables.dc_key]
        powers = center_poly.powers_
        coefs = center_model.coef_
        point = np.array([x, y, z])

        fourth_tensor = np.zeros((3, 3, 3, 3))

        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for l in range(3):
                        d4p = 0.0
                        for m, power in enumerate(powers):
                            counts = [i, j, k, l]
                            if any(
                                power[axis] < counts.count(axis) for axis in range(3)
                            ):
                                continue

                            coeff = coefs[m]
                            for axis in range(3):
                                for c in range(counts.count(axis)):
                                    coeff *= power[axis] - c

                            term = coeff * np.prod(
                                [
                                    point[n] ** (power[n] - counts.count(n))
                                    for n in range(3)
                                ]
                            )
                            d4p += term
                        fourth_tensor[i, j, k, l] = d4p

        return fourth_tensor

    def get_eq_U_hessian(self, num_ions):
        if num_ions not in self.ion_equilibrium_positions:
            self.find_equilib_position_single(num_ions)
        ionpositions = self.ion_equilibrium_positions[num_ions]

        n = len(ionpositions)
        Hessian = np.zeros((3 * n, 3 * n))
        for ion in range(n):
            xi, yi, zi = ionpositions[ion]
            for jon in range(n):
                xj, yj, zj = ionpositions[jon]
                dist_ij = np.linalg.norm(ionpositions[ion] - ionpositions[jon])
                dist_ij = max(dist_ij, 1e-28)
                for a in range(3):
                    for b in range(3):
                        if ion == jon:

                            if a == b:
                                d2Utrap = 0
                                d2Utrap = (
                                    constants.ion_charge
                                    * self.evaluate_center_poly_2ndderivatives(
                                        xi, yi, zi
                                    )[a][b]
                                )

                                d2Ucolmb = 0
                                for kon in range(n):
                                    if kon == ion:
                                        continue
                                    xk, yk, zk = ionpositions[kon]
                                    dist_ik = np.linalg.norm(
                                        ionpositions[ion] - ionpositions[kon]
                                    )
                                    d2Ucolmb += -1 / dist_ik**3
                                    d2Ucolmb += (
                                        3
                                        * (ionpositions[ion][a] - ionpositions[kon][a])
                                        ** 2
                                    ) / (dist_ik**5)
                                d2Ucolmb *= (
                                    constants.coulomb_constant * constants.ion_charge**2
                                )  # /2

                                Hessian[3 * ion + a, 3 * ion + b] += d2Ucolmb + d2Utrap

                            else:  # a != b
                                d2Utrap = 0
                                d2Utrap = (
                                    constants.ion_charge
                                    * self.evaluate_center_poly_2ndderivatives(
                                        xi, yi, zi
                                    )[a][b]
                                )

                                d2Ucolmb = 0
                                for kon in range(n):
                                    if kon == ion:
                                        continue
                                    xk, yk, zk = ionpositions[kon]
                                    dist_ik = np.linalg.norm(
                                        ionpositions[ion] - ionpositions[kon]
                                    )
                                    d2Ucolmb += (
                                        3
                                        * (ionpositions[ion][a] - ionpositions[kon][a])
                                        * (ionpositions[ion][b] - ionpositions[kon][b])
                                    ) / (dist_ik**5)
                                d2Ucolmb *= (
                                    constants.coulomb_constant * constants.ion_charge**2
                                )  # /2

                                Hessian[3 * ion + a, 3 * ion + b] += d2Ucolmb + d2Utrap

                        else:  # ion != jon

                            if a == b:
                                d2Utrap = 0

                                d2Ucolmb = 0
                                d2Ucolmb += 1 / dist_ij**3
                                d2Ucolmb += -(
                                    3
                                    * (ionpositions[ion][a] - ionpositions[jon][a]) ** 2
                                ) / (dist_ij**5)
                                d2Ucolmb *= (
                                    constants.coulomb_constant * constants.ion_charge**2
                                )  # /2

                                Hessian[3 * ion + a, 3 * jon + b] += d2Ucolmb + d2Utrap

                            else:  # a != b
                                d2Utrap = 0

                                d2Ucolmb = 0
                                d2Ucolmb += -(
                                    3
                                    * (ionpositions[ion][a] - ionpositions[jon][a])
                                    * (ionpositions[ion][b] - ionpositions[jon][b])
                                ) / (dist_ij**5)
                                d2Ucolmb *= (
                                    constants.coulomb_constant * constants.ion_charge**2
                                )  # /2

                                Hessian[3 * ion + a, 3 * jon + b] += d2Ucolmb + d2Utrap
        return Hessian

    def get_eq_3rd_der_tensor(self, num_ions):
        ionpositions = self.ion_equilibrium_positions[num_ions]
        n = len(ionpositions)

        # The final 3rd-derivative tensor, shape (3n, 3n, 3n)
        Tensor3 = np.zeros((3 * n, 3 * n, 3 * n))

        # --- 1) Add trap 3rd derivatives (diagonal in ion index) ---
        for i in range(n):
            # Evaluate the 3rd partials of the trap at (x_i, y_i, z_i)
            D3 = self.evaluate_center_poly_3rdderivatives(*ionpositions[i])
            # Multiply by ion charge if your definition requires it
            D3 *= constants.ion_charge

            # Insert into Tensor3 for all coordinate triples (a,b,c)
            for a in range(3):
                for b in range(3):
                    for c in range(3):
                        Tensor3[3 * i + a, 3 * i + b, 3 * i + c] += D3[a, b, c]

        # --- 2) Add Coulomb 3rd derivatives for all pairs p<q ---
        # Potential is sum_{p<q} 1/|r_p - r_q|.
        coul_pref = constants.coulomb_constant * (constants.ion_charge**2)

        # Pre-define the 8 possible ways to distribute partial derivatives to p or q
        # Each tuple is: ( (ion1, coord1), (ion2, coord2), (ion3, coord3), (s1, s2, s3) )
        # meaning partial #1 is w.r.t. x_{ion1, coord1}, partial #2 is w.r.t. x_{ion2, coord2}, etc.
        #
        # For example, ( (p,a), (p,b), (q,c), (+1, +1, -1) ) means:
        # partial wrt x_{p,a}, x_{p,b}, x_{q,c}, so the r-vector sign is (+1, +1, -1).
        #
        # We'll fill in the T3 entry accordingly: T3[3p+a, 3p+b, 3q+c] += ...
        from itertools import product

        # Because we have triple partial derivatives, we want all combinations of
        # "this derivative is on p or q" => 2 x 2 x 2 = 8 combos.
        # Then for each partial derivative we specify which (p or q) and which coordinate index (a,b,c).
        # We'll build these 8 combos in a loop to keep it super explicit/correct:

        partial_combos = []
        # sign +1 => partial wrt p, sign -1 => partial wrt q
        # We define them in a consistent order, but any order is fine as long as we fill the correct T3 indices.

        # For derivative #1, #2, #3 => choose (p or q).  We'll store sign +1 if p, -1 if q.
        for s1, s2, s3 in product([+1, -1], repeat=3):
            # We’ll figure out which ion's coordinate each partial corresponds to:
            # If s1 == +1 => partial wrt x_{p, a}, else wrt x_{q, a}, etc.
            # But we haven't bound 'a,b,c' yet. We'll do that in an inner loop.
            # So partial_combos will be a function of (s1, s2, s3).
            partial_combos.append((s1, s2, s3))

        def third_deriv_1overr(ra, rb, rc, a, b, c, r):
            """
            Given signed components (ra, rb, rc) = s1*r_vec[a], s2*r_vec[b], s3*r_vec[c],
            returns the 3rd derivative ∂^3/∂(x_a)∂(x_b)∂(x_c)(1/r) in 3D.

            The formula is:
                15*(ra*rb*rc)/r^7
                - 3 * [δ_{ab} rc + δ_{ac} rb + δ_{bc} ra ] / r^5
            where δ_{ab} is Kronecker delta = 1 if a==b else 0.
            """
            val = -15.0 * (ra * rb * rc) / (r**7)

            # +3 [ δ_{ab} rc + δ_{ac} rb + δ_{bc} ra ] / r^5
            if a == b:
                val += 3.0 * (rc / (r**5))
            if a == c:
                val += 3.0 * (rb / (r**5))
            if b == c:
                val += 3.0 * (ra / (r**5))

            return val

        for p in range(n):
            for q in range(p + 1, n):
                r_vec = ionpositions[p] - ionpositions[q]
                r = np.linalg.norm(r_vec)
                r = max(r, 1e-28)

                # For each triple of coordinate indices (a, b, c) = 0..2
                for a in range(3):
                    for b in range(3):
                        for c in range(3):
                            # Extract the "bare" (x_p - x_q) components
                            r_a = r_vec[a]
                            r_b = r_vec[b]
                            r_c = r_vec[c]

                            # 8 combos for distributing partials to (p or q)
                            for s1, s2, s3 in partial_combos:
                                # Compute the sign-adjusted components
                                # e.g., if s1=+1 => partial wrt x_{p,a}, if s1=-1 => partial wrt x_{q,a}

                                sign = s1 * s2 * s3

                                # The actual 3rd derivative:
                                d3_val = sign * third_deriv_1overr(
                                    r_a, r_b, r_c, a, b, c, r
                                )
                                d3_val *= (
                                    coul_pref  # multiply by Coulomb constant factor
                                )

                                # Figure out which [I, J, K] index in Tensor3 to add it to.
                                #
                                # partial #1 => (ion1, coord1) depends on s1: +1 => (p,a), -1 => (q,a)
                                if s1 == +1:
                                    i1 = 3 * p + a
                                else:
                                    i1 = 3 * q + a

                                # partial #2 => (ion2, coord2)
                                if s2 == +1:
                                    i2 = 3 * p + b
                                else:
                                    i2 = 3 * q + b

                                # partial #3 => (ion3, coord3)
                                if s3 == +1:
                                    i3 = 3 * p + c
                                else:
                                    i3 = 3 * q + c

                                Tensor3[i1, i2, i3] += d3_val

        return Tensor3

    def get_eq_4th_der_tensor(self, num_ions):
        # Get the positions of ions for the given num_ions key
        ionpositions = self.ion_equilibrium_positions[num_ions]
        n = len(ionpositions)

        # Initialize the fourth-order tensor: shape (3n, 3n, 3n, 3n)
        Tensor4 = np.zeros((3 * n, 3 * n, 3 * n, 3 * n))

        # -----------------------------------------------------------------
        # 1) Trap Contribution (diagonal blocks: only when all partials are wrt same ion)
        # -----------------------------------------------------------------
        for i in range(n):
            # Evaluate the trap's 4th derivatives at ion i; expected shape (3,3,3,3)
            D4_trap = self.evaluate_center_poly_4thderivatives(*ionpositions[i])
            # Multiply by the ion charge if required by your definition
            D4_trap *= constants.ion_charge
            for a in range(3):
                for b in range(3):
                    for c in range(3):
                        for d in range(3):
                            Tensor4[
                                3 * i + a, 3 * i + b, 3 * i + c, 3 * i + d
                            ] += D4_trap[a, b, c, d]

        # -----------------------------------------------------------------
        # 2) Coulomb Contribution (for each pair p < q)
        # -----------------------------------------------------------------
        coul_pref = constants.coulomb_constant * (constants.ion_charge**2)

        # Define the analytic fourth derivative of 1/r (with r already sign-adjusted)
        def fourth_deriv_1overr(ra, rb, rc, rd, a, b, c, d, r):
            # r: norm of the difference vector (>= 1e-28 to avoid div-by-zero)
            # ra, rb, rc, rd are already the sign-adjusted components.
            term1 = 105.0 * (ra * rb * rc * rd) / (r**9)
            term2 = 0.0
            # Six terms: note that δ_{ij}=1 if indices are equal else 0.
            if a == b:
                term2 += rc * rd
            if a == c:
                term2 += rb * rd
            if a == d:
                term2 += rb * rc
            if b == c:
                term2 += ra * rd
            if b == d:
                term2 += ra * rc
            if c == d:
                term2 += ra * rb
            term2 = 15.0 * term2 / (r**7)
            term3 = 0.0
            if a == b and c == d:
                term3 += 1.0
            if a == c and b == d:
                term3 += 1.0
            if a == d and b == c:
                term3 += 1.0
            term3 = 3.0 * term3 / (r**5)
            return term1 - term2 + term3

        # Loop over all pairs p < q
        for p in range(n):
            for q in range(p + 1, n):
                # Compute r_vec from ion p to ion q and its norm, with a floor to avoid div-by-zero.
                r_vec = ionpositions[p] - ionpositions[q]
                r = np.linalg.norm(r_vec)
                r = max(r, 1e-28)

                # Loop over all coordinate indices for the fourth derivative
                for a in range(3):
                    for b in range(3):
                        for c in range(3):
                            for d in range(3):
                                # Bare components (from r_vec)
                                r_a = r_vec[a]
                                r_b = r_vec[b]
                                r_c = r_vec[c]
                                r_d = r_vec[d]
                                # For the 4th derivative, there are 16 combinations of
                                # assigning each partial derivative to ion p or ion q.
                                for s1, s2, s3, s4 in product([+1, -1], repeat=4):
                                    # The sign factors: +1 means derivative with respect to ion p,
                                    # -1 means derivative with respect to ion q.

                                    sign = s1 * s2 * s3 * s4
                                    d4_val = sign * fourth_deriv_1overr(
                                        r_a, r_b, r_c, r_d, a, b, c, d, r
                                    )
                                    d4_val *= coul_pref

                                    # Map to global tensor indices:
                                    if s1 == +1:
                                        i1 = 3 * p + a
                                    else:
                                        i1 = 3 * q + a
                                    if s2 == +1:
                                        i2 = 3 * p + b
                                    else:
                                        i2 = 3 * q + b
                                    if s3 == +1:
                                        i3 = 3 * p + c
                                    else:
                                        i3 = 3 * q + c
                                    if s4 == +1:
                                        i4 = 3 * p + d
                                    else:
                                        i4 = 3 * q + d

                                    Tensor4[i1, i2, i3, i4] += d4_val
        return Tensor4

    # Used Checked1014
    def get_U_using_polyfit_dimensionless(self, ionpos_flat):
        """
        Dimensionless energy function: returns U / E_0
        where E_0 = q^2 / (4 pi eps_0 L_0)
        and positions are in units of L_0.
        """
        L_0 = constants.length_harmonic_approximation
        q = constants.ion_charge
        eps0 = constants.epsilon_0
        E_0 = q**2 / (4 * np.pi * eps0 * L_0)

        n = len(ionpos_flat) // 3
        pos_dim = ionpos_flat.reshape((n, 3))
        pos_SI = pos_dim * L_0  # convert to SI units for trap eval

        U_trap_scaled = 0.0
        for ion in range(n):
            x, y, z = pos_SI[ion]
            U_trap_scaled += q * self.evaluate_center_poly(x, y, z) / E_0  # volts
            # print(ion, U_trap,x,y,z)

        U_coulomb = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                r = np.linalg.norm(pos_dim[i] - pos_dim[j])
                if r < 1e-12:
                    r = 1e-8
                    # print("Warning: distance too small, setting to 1e-8")
                U_coulomb += 1.0 / r  # dimensionless

        # print("U_trap_scaled:", U_trap_scaled)
        # print("U_coulomb:", U_coulomb * L_0)
        # print("")
        # print("U_final:", U_trap_scaled + U_coulomb * L_0)
        return U_trap_scaled + U_coulomb

    # Used Checked1014
    def get_U_Grad_using_polyfit_dimensionless(self, ionpos_flat):
        """
        Returns dimensionless gradient of total potential energy: grad(U / E_0)
        Input: ionpos_flat in units of L_0
        Output: gradient in units of E_0 / L_0 (normalized force)
        """
        L_0 = constants.length_harmonic_approximation
        q = constants.ion_charge
        eps0 = constants.epsilon_0
        E_0 = q**2 / (4 * np.pi * eps0 * L_0)

        n = len(ionpos_flat) // 3
        pos_dim = ionpos_flat.reshape((n, 3))
        pos_SI = pos_dim * L_0  # convert to SI units for trap eval
        grad = np.zeros_like(pos_dim)

        # Coulomb gradient (dimensionless)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue

                diff = pos_dim[i] - pos_dim[j]

                r_squared = np.dot(diff, diff)
                r = np.sqrt(r_squared)

                if r < 1e-12:
                    grad[i] += diff / (1e-12**3)

                else:
                    grad[i] += diff / (r**3)

        # Trap gradient (evaluate using SI positions)
        for i in range(n):
            x, y, z = pos_SI[i]
            dx, dy, dz = self.evaluate_center_poly_1stderivatives(x, y, z)  # volts/m
            grad[i] += (q * L_0 / E_0) * np.array([dx, dy, dz])

        # print("grad norm:", np.linalg.norm(grad.flatten()))
        return grad.flatten()

    # Used
    def find_U_minimum(self, num_ions, intial_guess=None):
        """
        Minimizes the total potential energy (trap potential + Coulomb)
        for 'number_of_ions' ions.
        """

        if intial_guess is None:
            init_guess_flat = np.array(
                constants.ion_locations_intial_guess[num_ions]
            ).flatten()
            init_guess_flat += np.random.uniform(
                -1.0e-6, 1.0e-6, size=init_guess_flat.shape
            )
        if intial_guess is not None:
            init_guess_flat = intial_guess.flatten()
            # print("Initial guess:", init_guess_flat)
            # print("Initial guess shape:", init_guess_flat.shape)

        bounds = constants.ion_locations_bounds[num_ions]

        result = minimize(
            self.get_U_using_polyfit_dimensionless,  # <-- use dimensionless U
            init_guess_flat,
            method="L-BFGS-B",
            jac=self.get_U_Grad_using_polyfit_dimensionless,  # <-- pass the gradient
            bounds=bounds,
            options={
                # "gtol": 1e-20,
                "gtol": 1e-7,  # tighter gradient tolerance
                "ftol": 1e-7,  # tighter func change threshold
                "disp": False,
                "maxiter": 10000,
                "maxfun": 100000,
            },
        )
        # print("Exit message:", result.message)
        # print("Exit status code:", result.status)
        # print("Final gradient norm:", np.linalg.norm(result.jac))
        # print("Final function value:", result.fun)
        # print("Number of iterations:", result.nit)
        # print("Function evaluations:", result.nfev)
        # print("Max |grad| component:", np.max(np.abs(result.jac)))
        # print(" * * * * * * * * * * * * * * * * * * * * * * *")
        # if not result.success:
        #     print(
        #         "Minimization failed * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *:",
        #         result.message,
        #     )
        #     print(result)
        # else:
        #     pass
        #     print("Minimization successful!")
        #     print("Final potential energy:", result.fun)

        # return just the minimized positions
        return result.x.reshape((num_ions, 3)), result.fun

    # Used
    def find_U_minimum_robust(self, num_ions):
        init_guess = constants.ion_locations_intial_guess[num_ions]
        init_guess = np.array(init_guess).flatten()
        # print("Initial guess for ion positions:", init_guess)
        # U_test = self.get_U_using_polyfit_dimensionless(init_guess)
        U_min = np.inf
        U_eq = None
        for i in range(2):
            U_eqi, U_fini = self.find_U_minimum(num_ions)
            if U_fini < U_min:
                U_min = U_fini
                U_eq = U_eqi
        eq, u = self.find_U_minimum(num_ions, U_eq)
        return eq

    def find_equilib_position_single_normal(self, num_ions):
        for i in range(5):
            print("HI there the normal eq position thing is being run, are you sure it should be?")
        if num_ions not in self.ion_equilibrium_positions:
            print(f"Finding Umin for {num_ions} ions")
            eq_dimless = self.find_U_minimum_robust(num_ions)  # shape (n,3), unitless
            eq_SI = eq_dimless * constants.length_harmonic_approximation
            self.ion_equilibrium_positions[num_ions] = eq_SI

    # Used
    def find_all_equilib_positions_normal(self):
        """
        Finds the Umin for all ions.
        """
        for num_ions in range(1, constants.max_ion_in_chain + 1):
            print(f"Finding Umin for {num_ions} ions")
            eq_dimless = self.find_U_minimum_robust(num_ions)  # shape (n,3), unitless
            eq_SI = eq_dimless * constants.length_harmonic_approximation
            self.ion_equilibrium_positions[num_ions] = eq_SI
            # print("eq (m):", eq_SI)
            # print(
            #     "eq / L0 (dimensionless):",
            #     eq_dimless,
            # )

    def find_equilib_position_single_initialguess(self, num_ions):
        eq_dimless = np.array(constants.ion_locations_intial_guess[num_ions])
        eq_SI = eq_dimless * constants.length_harmonic_approximation
        self.ion_equilibrium_positions[num_ions] = eq_SI
        return eq_dimless

    def find_all_equilib_positions_initialguess(self):
        """
        Returns initial-guess positions for all ion counts.
        """
        out = {}
        for num_ions in range(1, constants.max_ion_in_chain + 1):
            eq_dimless = np.array(constants.ion_locations_intial_guess[num_ions])
            eq_SI = eq_dimless * constants.length_harmonic_approximation
            self.ion_equilibrium_positions[num_ions] = eq_SI
            out[num_ions] = eq_dimless
        return out

    def find_equilib_position_single_dummy1(self, num_ions):
        return self.find_equilib_position_single_initialguess(num_ions)

    def find_all_equilib_positions_dummy1(self):
        return self.find_all_equilib_positions_initialguess()

    def find_equilib_position_single_dummy2(self, num_ions):
        return self.find_equilib_position_single_initialguess(num_ions)

    def find_all_equilib_positions_dummy2(self):
        return self.find_all_equilib_positions_initialguess()

    def find_equilib_position_single_dummy3(self, num_ions):
        return self.find_equilib_position_single_initialguess(num_ions)

    def find_all_equilib_positions_dummy3(self):
        return self.find_all_equilib_positions_initialguess()

    def find_equilib_position_single(self, num_ions, minimizertype="Normal"):
        if minimizertype == "Normal":
            return self.find_equilib_position_single_normal(num_ions)
        if minimizertype == "InitGuess":
            return self.find_equilib_position_single_initialguess(num_ions)
        if minimizertype == "Dummy1":
            return self.find_equilib_position_single_dummy1(num_ions)
        if minimizertype == "Dummy2":
            return self.find_equilib_position_single_dummy2(num_ions)
        if minimizertype == "Dummy3":
            return self.find_equilib_position_single_dummy3(num_ions)
        else:
            return None

    def find_all_equilib_positions(self, minimizertype="Normal"):
        if minimizertype == "Normal":
            return self.find_all_equilib_positions_normal()
        if minimizertype == "InitGuess":
            return self.find_all_equilib_positions_initialguess()
        if minimizertype == "Dummy1":
            return self.find_all_equilib_positions_dummy1()
        if minimizertype == "Dummy2":
            return self.find_all_equilib_positions_dummy2()
        if minimizertype == "Dummy3":
            return self.find_all_equilib_positions_dummy3()
        else:
            return None


# #### POSIBLY UNUSED OR OLD ############################

# # unused?
# def get_U_using_polyfit(self, ionpos_flat):
#     n = len(ionpos_flat) // 3
#     positions = ionpos_flat.reshape((n, 3))

#     U_trap = 0.0
#     U_coulomb = 0.0

#     for ion in range(n):
#         x, y, z = positions[ion]

#         # Evaluate the model
#         voltage_at_point = self.evaluate_center_poly(x, y, z)

#         U_trap += voltage_at_point * constants.ion_charge

#     epsilon = 1e-28  # small buffer to avoid division-by-zero issues
#     for i in range(n):
#         for j in range(n):
#             if i == j:
#                 continue
#             dist = np.linalg.norm(positions[i] - positions[j])
#             dist = max(dist, epsilon)
#             U_coulomb += (
#                 constants.coulomb_constant * constants.ion_charge**2
#             ) / dist

#     return (U_trap + U_coulomb / 2) * 1e25

# # unused?
# def get_U_using_polyfit_zeroed(self, ionpos_flat):
#     n = len(ionpos_flat) // 3
#     inital_guess = constants.ion_locations_intial_guess[n]
#     initial_guess = np.array(inital_guess).flatten()

#     return self.get_U_using_polyfit(ionpos_flat) - self.get_U_using_polyfit(
#         initial_guess
#     )

# # unused?
# def get_U_Grad_using_polyfit(self, ionpos_flat):
#     """
#     Ok we are going to re-write this such that the gradient is calcualted as if the U is set to zero at the initial_guess
#     """
#     n = len(ionpos_flat) // 3
#     positions = ionpos_flat.reshape((n, 3))
#     GradU = np.zeros(3 * n)

#     for i in range(n):
#         xi, yi, zi = positions[i]
#         dU_dxi, dU_dyi, dU_dzi = self.evaluate_center_poly_1stderivatives(
#             xi, yi, zi
#         )
#         GradU[3 * i] += dU_dxi * constants.ion_charge
#         GradU[3 * i + 1] += dU_dyi * constants.ion_charge
#         GradU[3 * i + 2] += dU_dzi * constants.ion_charge

#         for j in range(n):
#             if i == j:
#                 continue
#             xj, yj, zj = positions[j]
#             dist = np.linalg.norm(positions[i] - positions[j])
#             dist = max(dist, 1e-12)

#             GradU[3 * i] += (
#                 (constants.coulomb_constant * constants.ion_charge**2)
#                 * (xj - xi)
#                 / (dist) ** (3 / 2)
#             )

#             GradU[3 * i + 1] += (
#                 (constants.coulomb_constant * constants.ion_charge**2)
#                 * (yj - yi)
#                 / (dist) ** (3 / 2)
#             )

#             GradU[3 * i + 2] += (
#                 (constants.coulomb_constant * constants.ion_charge**2)
#                 * (zj - zi)
#                 / (dist) ** (3 / 2)
#             )

#     return GradU * 1e25

# # failed attempt
# def find_U_minimum_basin_hopping(self, num_ions, inital_guess=None):
#     """
#     Minimizes the total potential energy (trap potential + Coulomb)
#     for 'number_of_ions' ions using basin hopping.
#     """
#     if inital_guess is None:
#         init_guess_flat = np.array(
#             constants.ion_locations_intial_guess[num_ions]
#         ).flatten()
#     else:
#         init_guess_flat = inital_guess.flatten()

#     bounds = constants.ion_locations_bounds[num_ions]

#     result = basinhopping(
#         self.get_U_using_polyfit_zeroed,
#         init_guess_flat,
#         niter=100,
#         T=1.0,
#         stepsize=0.5,
#         minimizer_kwargs={
#             "method": "L-BFGS-B",
#             "jac": self.get_U_Grad_using_polyfit,
#             "bounds": bounds,
#             "options": {"disp": True},
#         },
#     )

#     if not result.success:
#         print("Minimization failed:", result.message)
#         # print(result)
#     else:
#         print("Minimization successful!")
#         print("Final potential energy:", result.fun)

#     # return just the minimized positions
#     return result.x.reshape((num_ions, 3))

# # huh?
# def check_grad(self):
#     ionpos_flat = np.array(constants.ion_locations_intial_guess[5]).flatten()
#     print(
#         check_grad(
#             lambda x: self.get_U_using_polyfit_dimensionless(x),
#             lambda x: self.get_U_Grad_using_polyfit_dimensionless(x),
#             ionpos_flat,
#         )
#     )
