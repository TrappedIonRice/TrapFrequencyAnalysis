# sim/voltage_interface.py
from __future__ import annotations
import time
from typing import Tuple
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import constants
from trapping_variables import drive_colname  # single source of truth for column names



class VoltageInterfaceMixin:
    """
    Methods that *read* voltage data from the grid/DF to compute values/gradients/minima.
    Expects `self` to have:
      - self.total_voltage_df : pandas.DataFrame with columns x,y,z and per-drive TotalV columns
      - self.trapVariables    : Trapping_Vars (for drive_colname(dc_key) aka 'Static_TotalV')
    """

    def find_V_min(self, step_size=5):
        """
        Finds and returns the point with the minimum total voltage.

        To catch errors, the minimum 100 points are found. If they are all close to each other, then the minimum is found.
        If there are outliers, they are thrown out, then the minimum is found.
        then the points around this min are fit to a quadratic to find the best fit min

        step_size is used to determine the cutout size for the R3 fit around the minimum point.

        args:
            step_size (float): The step size(in microns) to use for the cutout around the minimum point. Default is 5 microns.
            Note if step size is too small an error will be thrown (must be over 5)

        returns:
            the best fit minimum point (x,y,z) in meters and the dataframe minimum
        """

        if step_size <= 4.9:
            raise ValueError("Step size must be greater than 5.")

        step_size = step_size * 1e-6  # Convert microns to meters for calculations

        if self.total_voltage_df is None:
            print("Total voltage data not available.")
            return None
        time1 = time.time()

        # Sort by TotalV to find the minimum values

        def find_nsmallest_df(df, colname, n=100):
            # 1) Extract column values
            arr = df[colname].to_numpy()

            # 2) Grab the indices of the n smallest values (unordered)
            idx = np.argpartition(arr, n)[:n]

            # 3) Sort those n rows by their actual values so the final result is ascending
            idx_sorted = idx[np.argsort(arr[idx])]

            # 4) Index back into the DataFrame
            return df.iloc[idx_sorted]

        # Usage:
        sorted_df = find_nsmallest_df(self.total_voltage_df, "TotalV", n=100)

        # Check proximity of the top 1000 minimum points
        points = sorted_df[["x", "y", "z"]].values
        calcV_values = sorted_df["TotalV"].values

        # Calculate distances between points
        distances = np.linalg.norm(points[:, np.newaxis] - points, axis=2)

        # Calculate average distance from each point to all other points
        average_distances = np.mean(distances, axis=1)

        # Identify outliers based on average distance threshold
        threshold = np.percentile(average_distances, 80)
        outliers_mask = average_distances > threshold

        # Filter out outliers
        filtered_points = points[~outliers_mask]
        filtered_calcV = calcV_values[~outliers_mask]

        if len(filtered_points) < 70:
            print("Many min points removed")
            print("Total points removed: " + str(len(points) - len(filtered_points)))

        # Find the minimum point among the filtered points
        min_index = np.argmin(filtered_calcV)
        min_point = filtered_points[min_index]

        ## Get the surrounding points for R3 fit using stepsize as the cutoff
        cutout_of_df = self.total_voltage_df[
            (
                self.total_voltage_df["x"].between(
                    min_point[0] - (5 * step_size), min_point[0] + (5 * step_size)
                )
            )
            & (
                self.total_voltage_df["y"].between(
                    min_point[1] - step_size, min_point[1] + step_size
                )
            )
            & (
                self.total_voltage_df["z"].between(
                    min_point[2] - step_size, min_point[2] + step_size
                )
            )
        ]

        voltage_vals = cutout_of_df["TotalV"].values
        xyz_vals_uncentered = cutout_of_df[["x", "y", "z"]].values

        # Make the Point of interest the origin (0,0,0) and move the other points accordingly
        xyz_vals_centered = xyz_vals_uncentered - min_point

        poly = PolynomialFeatures(degree=2, include_bias=True)
        X_poly = poly.fit_transform(xyz_vals_centered)

        # Fit the model
        model = LinearRegression()
        model.fit(X_poly, voltage_vals)

        # (4) Extract coefficients
        # model.coef_ is length-10 if include_bias=True for 3D data; also consider model.intercept_
        c0 = model.intercept_
        c1, c2, c3, c4, c5, c6, c7, c8, c9 = model.coef_[
            1:
        ]  # skipping the bias column's coef

        # (5) Solve gradient=0 for (x, y, z) in the centered frame
        H = np.array([[2 * c4, c5, c6], [c5, 2 * c7, c8], [c6, c8, 2 * c9]])
        linear_terms = np.array([c1, c2, c3])

        delta_xyz_centered = np.linalg.solve(H, -linear_terms)

        # (6) Shift back to original coordinates
        best_fit_minimum = min_point + delta_xyz_centered

        time5 = time.time()

        # print("Total time taken to find min: ", time5 - time1)

        # print("best_fit_minimum: ", best_fit_minimum)
        # print("min_point: ", min_point)
        return best_fit_minimum, min_point

    def find_V_trap_at_point_fast_and_dirty(self, x, y, z, starting_step=0.49):
        """
        Lol dont use this

        Finds and returns the potential of the trap at a given point (x,y,z).
        The function takes in the coordinates of the desired point in space and returns the potential (V).

        We do this fast by using the df to find the 8 closest points and avergaing them.
        """
        # starting_step = starting_step * 1e-6  # Convert microns to meters for calculations

        ## Get the surrounding points for R3 fit using stepsize as the cutoff
        cutout_of_df = pd.DataFrame()

        # while len(cutout_of_df) < 10:
        #     cutout_of_df = self.total_voltage_df[
        #     (self.total_voltage_df["x"].between(x - (20 * starting_step), x + (20 * starting_step)))
        #     & (self.total_voltage_df["y"].between(y - starting_step, y + starting_step))
        #     & (self.total_voltage_df["z"].between(z - starting_step, z + starting_step))
        #     ]

        #     starting_step += (.01 * 1e-6)
        # print(starting_step, "************************************************************************************")
        starting_step = (
            starting_step * 1e-6
        )  # Convert microns to meters for calculations
        cutout_of_df = self.total_voltage_df[
            (
                self.total_voltage_df["x"].between(
                    x - (20 * starting_step), x + (20 * starting_step)
                )
            )
            & (self.total_voltage_df["y"].between(y - starting_step, y + starting_step))
            & (self.total_voltage_df["z"].between(z - starting_step, z + starting_step))
        ]

        voltage_vals = cutout_of_df["TotalV"].values
        print(len(voltage_vals))
        avg_V = np.mean(voltage_vals)

        return avg_V

    def find_V_trap_at_point(self, x, y, z, starting_step=0.4, derivs=False):
        """
        Finds and returns the potential of the trap at a given point (x,y,z).
        The function takes in the coordinates of the desired point in space and returns the potential (V).

        To do this we find the closest 50 values in the dataframe to the point (x,y,z) and then use these points to find the potential at the point.
        """

        starting_step = (
            starting_step * 1e-6
        )  # Convert microns to meters for calculations

        ## Get the surrounding points for R3 fit using stepsize as the cutoff
        cutout_of_df = pd.DataFrame()

        len_df = len(self.total_voltage_df)

        while len(cutout_of_df) < 20:
            cutout_of_df = self.total_voltage_df[
                (
                    self.total_voltage_df["x"].between(
                        x - (10 * starting_step), x + (10 * starting_step)
                    )
                )
                & (
                    self.total_voltage_df["y"].between(
                        y - starting_step, y + starting_step
                    )
                )
                & (
                    self.total_voltage_df["z"].between(
                        z - starting_step, z + starting_step
                    )
                )
            ]

            starting_step += 5 * 1e-6

        voltage_vals = cutout_of_df["TotalV"].values
        # print(len(cutout_of_df), " points found in cutout")
        # print("voltage_vals: ", voltage_vals)
        xyz_vals_uncentered = cutout_of_df[["x", "y", "z"]].values

        # Make the Point of interest the origin (0,0,0) and move the other points accordingly
        xyz_vals_centered = xyz_vals_uncentered - [x, y, z]
        t1 = time.time()

        poly = PolynomialFeatures(degree=3, include_bias=True)
        X_poly = poly.fit_transform(xyz_vals_centered)

        # Fit the model
        model = LinearRegression()
        model.fit(X_poly, voltage_vals)
        t2 = time.time()

        # find and print out the r2 of the fit
        r2 = model.score(X_poly, voltage_vals)
        # print("R-squared of the fit:", r2)

        # Get derivatives (d/dx, d/dy, d/dz) at point
        derivatives = model.coef_[1:4]

        # find the value of the fit at the origin
        Vvalue_at_point = model.predict(poly.transform([[0, 0, 0]]))
        # print("Time taken to find V at point: ", t2 - t1)

        if derivs:
            # Return the derivatives at the point
            return Vvalue_at_point[0], derivatives
        return Vvalue_at_point[0]
        # Calculate the potential energy of the ions

    def plot_total_voltage_along_axis(
        self, axis: str, width_um: float, polyfit: int = 4
    ):
        """
        Plot TotalV (or Static_TotalV if present) along a chosen axis near the origin.

        Args:
            axis: 'x', 'y', or 'z'
            width_um: half-width along the chosen axis (microns)
            polyfit: polynomial degree for the fit (default 4)
        """
        axis = axis.lower().strip()
        if axis not in ("x", "y", "z"):
            raise ValueError("axis must be one of: 'x', 'y', 'z'")

        df = self.total_voltage_df
        if df is None or df.empty:
            raise ValueError("Total voltage dataframe is empty.")

        value_col = "Static_TotalV" if "Static_TotalV" in df.columns else "TotalV"
        if value_col not in df.columns:
            raise KeyError("No total voltage column found (Static_TotalV or TotalV).")

        # Use half-grid spacing as a tolerance for perpendicular axes
        tol = {}
        for ax in ("x", "y", "z"):
            uniq = np.sort(df[ax].unique())
            if len(uniq) < 2:
                tol[ax] = 0.0
            else:
                tol[ax] = 0.5 * float(np.min(np.diff(uniq)))

        width_m = float(width_um) * 1e-6
        filters = [df[axis].between(-width_m, width_m)]
        for ax in ("x", "y", "z"):
            if ax == axis:
                continue
            filters.append(df[ax].between(-tol[ax], tol[ax]))

        mask = np.logical_and.reduce(filters)
        cutout = df[mask].copy()
        if cutout.empty:
            raise ValueError("No points found on axis within the requested width.")

        cutout.sort_values(axis, inplace=True)
        axis_vals = cutout[axis].to_numpy()
        volt_vals = cutout[value_col].to_numpy()

        # Fit quadratic and quartic
        coeffs2 = np.polyfit(axis_vals, volt_vals, 2)
        poly2 = np.poly1d(coeffs2)
        coeffs4 = np.polyfit(axis_vals, volt_vals, 4)
        poly4 = np.poly1d(coeffs4)

        fig, ax = plt.subplots()
        ax.scatter(axis_vals * 1e6, volt_vals, s=10, alpha=0.7, label="data")
        xs = np.linspace(axis_vals.min(), axis_vals.max(), 400)
        ax.plot(xs * 1e6, poly2(xs), color="orange", label="deg2 fit")
        ax.plot(xs * 1e6, poly4(xs), color="red", label="deg4 fit")
        ax.set_xlabel(f"{axis} (um)")
        ax.set_ylabel(value_col)
        ax.set_title(f"{value_col} along {axis}-axis")
        ax.legend()

        eq2 = [f"{c:.3e}*x^{p}" for p, c in zip(range(2, -1, -1), coeffs2)]
        eq4 = [f"{c:.3e}*x^{p}" for p, c in zip(range(4, -1, -1), coeffs4)]
        print("Quadratic fit:")
        print(" + ".join(eq2))
        print("Quartic fit:")
        print(" + ".join(eq4))

        plt.show()

    def _secular_freq_from_second_derivative(self, second_derivative):
        """
        Convert second derivative (V/m^2) to secular frequency (MHz).
        """
        return (
            np.sign(second_derivative)
            * np.sqrt(
                (constants.ion_charge / constants.ion_mass) * abs(second_derivative)
            )
            / (2 * np.pi)
        )

    def plot_total_voltage_along_axes(self, width_um=[100, 20, 20]):

        if len(width_um) != 3:
            raise ValueError("width_um must be [xwidth, ywidth, zwidth]")

        df = self.total_voltage_df
        if df is None or df.empty:
            raise ValueError("Total voltage dataframe is empty.")

        required_cols = ["Static_TotalV", "Static_RFPseudoV", "Static_DCV"]
        for col in required_cols:
            if col not in df.columns:
                raise KeyError(f"{col} not found. Run update_total_voltage_columns().")

        widths = dict(zip(("x", "y", "z"), map(float, width_um)))

        fig, axes = plt.subplots(3, 3, figsize=(14, 10), constrained_layout=True)

        tol = {}
        for ax in ("x", "y", "z"):
            uniq = np.sort(df[ax].unique())
            tol[ax] = 0.0 if len(uniq) < 2 else 0.5 * float(np.min(np.diff(uniq)))

        potential_map = {0: "Static_TotalV", 1: "Static_RFPseudoV", 2: "Static_DCV"}
        row_titles = {0: "Total", 1: "RF only", 2: "DC only"}

        # store secular frequencies for ratio calculation
        sec_freq = {}

        for row in range(3):
            value_col = potential_map[row]

            for col, axis in enumerate(("x", "y", "z")):
                ax_plot = axes[row, col]
                width_m = widths[axis] * 1e-6

                filters = [df[axis].between(-width_m, width_m)]
                for other_ax in ("x", "y", "z"):
                    if other_ax != axis:
                        filters.append(df[other_ax].between(-tol[other_ax], tol[other_ax]))

                mask = np.logical_and.reduce(filters)
                cutout = df[mask].copy()
                if cutout.empty:
                    raise ValueError(f"No points found on {axis}-axis within width.")

                cutout.sort_values(axis, inplace=True)
                axis_vals = cutout[axis].to_numpy()
                volt_vals = cutout[value_col].to_numpy()

                # fits
                coeffs2 = np.polyfit(axis_vals, volt_vals, 2)
                coeffs4 = np.polyfit(axis_vals, volt_vals, 4)
                poly2 = np.poly1d(coeffs2)
                poly4 = np.poly1d(coeffs4)

                xs = np.linspace(axis_vals.min(), axis_vals.max(), 400)

                ax_plot.scatter(axis_vals * 1e6, volt_vals, s=10, alpha=0.7)
                ax_plot.plot(xs * 1e6, poly2(xs), color="orange", label="deg2 fit")
                ax_plot.plot(xs * 1e6, poly4(xs), color="red", label="deg4 fit")

                # quartic curvature at 0
                a, b, c, d, e = coeffs4
                d2 = 2.0 * c
                freq_hz = self._secular_freq_from_second_derivative(d2)
                sec_freq[axis] = freq_hz

                # label text
                label_lines = [f"d2V/d{axis}2 = {d2:.3e}", f"sec f = {freq_hz:.3e} Hz"]

                # add ratio sec f z/x for total potential row
                if row == 0 and all(k in sec_freq for k in ("x", "z")):
                    ratio = sec_freq["z"] / sec_freq["x"]
                    label_lines.append(f"sec f_z / sec f_x = {ratio:.5f}")

                # add q_vert for RF-only row
                if row == 1 and axis == "z":
                    q_vert = (2 * np.sqrt(2) * freq_hz) / ((36) * 10**6) # Change here the RF_Drive_Frequency
                    label_lines.append(f"q_vert = {q_vert:.5f}")

                ax_plot.text(
                    0.02,
                    0.98,
                    "\n".join(label_lines),
                    transform=ax_plot.transAxes,
                    va="top",
                    ha="left",
                    fontsize=8,
                    bbox={"boxstyle": "round", "fc": "white", "ec": "0.8", "alpha": 0.9},
                )

                if row == 0:
                    ax_plot.set_title(f"{axis}-axis")
                if col == 0:
                    ax_plot.set_ylabel(f"{row_titles[row]}\nPotential (V)")
                if row == 2:
                    ax_plot.set_xlabel(f"{axis} (µm)")

                ax_plot.grid(True)

        return fig

# This function here was a change to plot all the graphs for Total potential, DC only and RF only******************************************************

    # def plot_total_voltage_along_axes(self, width_um=[100, 20, 20]):
    # # """
    # # 3x3 grid:

    # #     Row 1 -> Total potential (x, y, z)
    # #     Row 2 -> RF pseudopotential (x, y, z)
    # #     Row 3 -> DC only (x, y, z)

    # # Includes quadratic & quartic fits and secular frequency extraction.
    # # """

    #     if len(width_um) != 3:
    #         raise ValueError("width_um must be [xwidth, ywidth, zwidth]")

    #     df = self.total_voltage_df
    #     if df is None or df.empty:
    #         raise ValueError("Total voltage dataframe is empty.")

    #     required_cols = ["Static_TotalV", "Static_RFPseudoV", "Static_DCV"]
    #     for col in required_cols:
    #         if col not in df.columns:
    #             raise KeyError(f"{col} not found. Run update_total_voltage_columns().")

    #     widths = dict(zip(("x", "y", "z"), map(float, width_um)))

    #     fig, axes = plt.subplots(3, 3, figsize=(14, 10), constrained_layout=True)

    #     # precompute tolerances
    #     tol = {}
    #     for ax in ("x", "y", "z"):
    #         uniq = np.sort(df[ax].unique())
    #         if len(uniq) < 2:
    #             tol[ax] = 0.0
    #         else:
    #             tol[ax] = 0.5 * float(np.min(np.diff(uniq)))

    #     potential_map = {
    #         0: "Static_TotalV",
    #         1: "Static_RFPseudoV",
    #         2: "Static_DCV",
    #     }

    #     row_titles = {
    #         0: "Total",
    #         1: "RF only",
    #         2: "DC only",
    #     }

    #     for row in range(3):
    #         value_col = potential_map[row]

    #         for col, axis in enumerate(("x", "y", "z")):
    #             ax_plot = axes[row, col]

    #             width_m = widths[axis] * 1e-6

    #             filters = [df[axis].between(-width_m, width_m)]
    #             for other_ax in ("x", "y", "z"):
    #                 if other_ax == axis:
    #                     continue
    #                 filters.append(df[other_ax].between(-tol[other_ax], tol[other_ax]))

    #             mask = np.logical_and.reduce(filters)
    #             cutout = df[mask].copy()

    #             if cutout.empty:
    #                 raise ValueError(f"No points found on {axis}-axis within width.")

    #             cutout.sort_values(axis, inplace=True)
    #             axis_vals = cutout[axis].to_numpy()
    #             volt_vals = cutout[value_col].to_numpy()

    #             # fits
    #             coeffs2 = np.polyfit(axis_vals, volt_vals, 2)
    #             coeffs4 = np.polyfit(axis_vals, volt_vals, 4)
    #             poly2 = np.poly1d(coeffs2)
    #             poly4 = np.poly1d(coeffs4)

    #             xs = np.linspace(axis_vals.min(), axis_vals.max(), 400)

    #             ax_plot.scatter(axis_vals * 1e6, volt_vals, s=10, alpha=0.7)
    #             ax_plot.plot(xs * 1e6, poly2(xs), color="orange", label="deg2 fit")
    #             ax_plot.plot(xs * 1e6, poly4(xs), color="red", label="deg4 fit")

    #             # quartic curvature at 0
    #             a, b, c, d, e = coeffs4
    #             d2 = 2.0 * c
    #             freq_hz = self._secular_freq_from_second_derivative(d2)

    #             ax_plot.text(
    #                 0.02,
    #                 0.98,
    #                 f"d2V/d{axis}2 = {d2:.3e}\nsec f = {freq_hz:.3e} Hz",
    #                 transform=ax_plot.transAxes,
    #                 va="top",
    #                 ha="left",
    #                 fontsize=8,
    #                 bbox={"boxstyle": "round", "fc": "white", "ec": "0.8", "alpha": 0.9},
    #             )

    #             if row == 0:
    #                 ax_plot.set_title(f"{axis}-axis")

    #             if col == 0:
    #                 ax_plot.set_ylabel(f"{row_titles[row]}\nPotential (V)")

    #             if row == 2:
    #                 ax_plot.set_xlabel(f"{axis} (µm)")

    #             ax_plot.grid(True)

    #     return fig
    
    #*******************************************************************************************************

    # def plot_total_voltage_along_axes(self, width_um = [1000,1000,1000]):
    #     """
    #     Plot TotalV along x, y, z axes and return the figure (no plt.show).

    #     Args:
    #         width_um: [xwidth, ywidth, zwidth] in microns
    #     """
    #     if len(width_um) != 3:
    #         raise ValueError("width_um must be [xwidth, ywidth, zwidth]")

    #     widths = {
    #         "x": float(width_um[0]),
    #         "y": float(width_um[1]),
    #         "z": float(width_um[2]),
    #     }

    #     fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)

    #     for ax_key, ax_plot in zip(("x", "y", "z"), axes):
    #         # use existing single-axis plotter but render on provided axis
    #         axis = ax_key
    #         df = self.total_voltage_df
    #         if df is None or df.empty:
    #             raise ValueError("Total voltage dataframe is empty.")

    #         value_col = "Static_TotalV" if "Static_TotalV" in df.columns else "TotalV"
    #         if value_col not in df.columns:
    #             raise KeyError(
    #                 "No total voltage column found (Static_TotalV or TotalV)."
    #             )

    #         tol = {}
    #         for ax in ("x", "y", "z"):
    #             uniq = np.sort(df[ax].unique())
    #             if len(uniq) < 2:
    #                 tol[ax] = 0.0
    #             else:
    #                 tol[ax] = 0.5 * float(np.min(np.diff(uniq)))

    #         width_m = widths[axis] * 1e-6
    #         filters = [df[axis].between(-width_m, width_m)]
    #         for ax in ("x", "y", "z"):
    #             if ax == axis:
    #                 continue
    #             filters.append(df[ax].between(-tol[ax], tol[ax]))

    #         mask = np.logical_and.reduce(filters)
    #         cutout = df[mask].copy()
    #         if cutout.empty:
    #             raise ValueError(f"No points found on {axis}-axis within width.")

    #         cutout.sort_values(axis, inplace=True)
    #         axis_vals = cutout[axis].to_numpy()
    #         volt_vals = cutout[value_col].to_numpy()

    #         # fits
    #         coeffs2 = np.polyfit(axis_vals, volt_vals, 2)
    #         coeffs4 = np.polyfit(axis_vals, volt_vals, 4)
    #         poly2 = np.poly1d(coeffs2)
    #         poly4 = np.poly1d(coeffs4)

    #         xs = np.linspace(axis_vals.min(), axis_vals.max(), 400)
    #         ax_plot.scatter(axis_vals * 1e6, volt_vals, s=10, alpha=0.7, label="data")
    #         ax_plot.plot(xs * 1e6, poly2(xs), color="orange", label="deg2 fit")
    #         ax_plot.plot(xs * 1e6, poly4(xs), color="red", label="deg4 fit")

    #         # second derivative from quartic at x=0: if coeffs4 = [a,b,c,d,e]
    #         # V(x) = a x^4 + b x^3 + c x^2 + d x + e => V''(0) = 2c
    #         a, b, c, d, e = coeffs4
    #         d2 = 2.0 * c
    #         freq_hz = self._secular_freq_from_second_derivative(d2)

    #         ax_plot.text(
    #             0.02,
    #             0.98,
    #             f"d2V/d{axis}2={d2:.3e}\nsec f={freq_hz:.3e} Hz",
    #             transform=ax_plot.transAxes,
    #             va="top",
    #             ha="left",
    #             fontsize=9,
    #             bbox={"boxstyle": "round", "fc": "white", "ec": "0.8", "alpha": 0.9},
    #     )            
    #         ax_plot.set_title(f"{value_col} along {axis}-axis")
    #         ax_plot.set_xlabel(f"{axis} (um)")
    #         ax_plot.set_ylabel(value_col)
    #         ax_plot.legend()


    #     return fig

    def plot_total_voltage_plane_cuts(self, n: int = 120, poly_deg: int | None = None):
        """
        Plot Static_TotalV fit plane cuts at x=0, y=0, z=0 and return the figure.
        """
        dc_key = getattr(self.trapVariables, "dc_key", None)
        fit = None
        if dc_key is not None and hasattr(self, "center_fits"):
            fit = self.center_fits.get(dc_key)

        if fit is None:
            if not hasattr(self, "update_center_polys"):
                raise RuntimeError("No center fit found and update_center_polys not available.")
            if poly_deg is None:
                poly_deg = 4
            self.update_center_polys(polyfit_deg=poly_deg)
            if dc_key is not None and hasattr(self, "center_fits"):
                fit = self.center_fits.get(dc_key)

        if fit is None:
            raise RuntimeError("Center fit for Static_TotalV not available.")

        model, poly, _r2 = fit

        span_x = constants.center_region_x_um * 1e-6
        span_y = constants.center_region_y_um * 1e-6
        span_z = constants.center_region_z_um * 1e-6

        def _eval_grid(xg, yg, zg):
            pts = np.c_[xg.ravel(), yg.ravel(), zg.ravel()]
            vals = model.predict(poly.transform(pts))
            return vals.reshape(xg.shape)

        x = np.linspace(-span_x, span_x, n)
        y = np.linspace(-span_y, span_y, n)
        z = np.linspace(-span_z, span_z, n)

        X_xy, Y_xy = np.meshgrid(x, y, indexing="xy")
        Z0 = np.zeros_like(X_xy)
        V_xy = _eval_grid(X_xy, Y_xy, Z0)

        X_xz, Z_xz = np.meshgrid(x, z, indexing="xy")
        Y0 = np.zeros_like(X_xz)
        V_xz = _eval_grid(X_xz, Y0, Z_xz)

        Y_yz, Z_yz = np.meshgrid(y, z, indexing="xy")
        X0 = np.zeros_like(Y_yz)
        V_yz = _eval_grid(X0, Y_yz, Z_yz)

        fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
        im0 = axes[0].contourf(X_xy * 1e6, Y_xy * 1e6, V_xy, levels=40, cmap="viridis")
        axes[0].set_title("Static_TotalV fit @ z=0")
        axes[0].set_xlabel("x (um)")
        axes[0].set_ylabel("y (um)")
        fig.colorbar(im0, ax=axes[0])

        im1 = axes[1].contourf(X_xz * 1e6, Z_xz * 1e6, V_xz, levels=40, cmap="viridis")
        axes[1].set_title("Static_TotalV fit @ y=0")
        axes[1].set_xlabel("x (um)")
        axes[1].set_ylabel("z (um)")
        fig.colorbar(im1, ax=axes[1])

        im2 = axes[2].contourf(Y_yz * 1e6, Z_yz * 1e6, V_yz, levels=40, cmap="viridis")
        axes[2].set_title("Static_TotalV fit @ x=0")
        axes[2].set_xlabel("y (um)")
        axes[2].set_ylabel("z (um)")
        fig.colorbar(im2, ax=axes[2])

        return fig
