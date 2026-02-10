"""
This file will contain the main base of the simulation class
"""

import math
import os
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import dataextractor
import constants
import time
import matplotlib.cm as cm
from multiprocessing import Pool, cpu_count
from joblib import Parallel, delayed
from scipy.interpolate import griddata
import numexpr as ne
from typing import Optional
from trapping_variables import Trapping_Vars, DriveKey
from scipy.optimize import minimize, BFGS, basinhopping
from scipy.optimize import differential_evolution
from scipy.interpolate import RBFInterpolator
from trapping_variables import Trapping_Vars, DriveKey, drive_colname

from sim.voltage_interfaceMixin import VoltageInterfaceMixin
from sim.voltage_fitsMixin import VoltageFitsMixin
from sim.Umin_ReqMixin import Umin_ReqMixin
from sim.StaticCoupolingMixin import StaticCoupolingMixin
from sim.StaticNormalModes_EigenMixin import StaticNormalModes_EigenMixin

from sim.simulation_ploting import sim_ploting
from sim.simulation_fitting import sim_normalfitting


class Simulation(
    Umin_ReqMixin,
    StaticNormalModes_EigenMixin,
    StaticCoupolingMixin,
    VoltageInterfaceMixin,
    VoltageFitsMixin,
    sim_ploting,
    sim_normalfitting,
):
    def __init__(self, dataset, trapVars=Trapping_Vars()):
        """
        Initialize the simulation object by making sure the data is all extracted and finding total voltages if given electodes

        dataset: the string of the desired data folder
        varaibles: and instance of the Electrode_vars class containing the desired electrode variables
        """
        timesimstart = time.time()
        print("initializing simulation")
        self.dataset = dataset
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.file_path = os.path.join(repo_root, "Data", dataset, "")


        # Hi
        self.trapVariables = trapVars

        self.total_voltage_df = None

        # get the dataframe if it exists, otherwise make it
        if os.path.exists(self.file_path + "combined_dataframe.csv"):
            self.total_voltage_df = pd.read_pickle(
                self.file_path + "combined_dataframe.csv"
            )
        else:
            print("making combined dataframe")
            print(self.file_path + "combined_dataframe.csv")
            self.total_voltage_df = dataextractor.make_simulation_dataframe(
                self.file_path
            )

        # get the total voltage at each point based on the electrode variables
        self.update_total_voltage_columns()

        # init other variables
        self.center_fits = {}  # Placeholder for the center fit model
        self.ion_equilibrium_positions = {}
        self.ion_eigenvectors = {}
        self.ion_eigenvalues = {}
        self.normal_modes_and_frequencies = {}
        self.driven_g_0_2_couplings = {}
        self.driven_g_0_3_couplings = {}
        self.inherent_g_0_3_couplings = {}
        self.inherent_g_0_4_couplings = {}

        self.principal_dirs = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        self.principal_dir_normalmodes_andfrequencies = {}

        timesimstop = time.time()
        print(
            "simulation initialized in " + str(timesimstop - timesimstart) + " seconds"
        )

    def get_trapping_variables(self, electrode):
        """
        Returns the electrode variables used by this simulation for a given electrode.
        """
        return self.trapVariables

    def update_total_voltage_columns(self):
        """
        Build total-voltage columns:
        - 'Static_TotalV' = DC scalar potential + pseudopotential from the fastest non-DC drive
        - '<drive>_TotalV' for each non-DC drive = scalar potential from that drive only
        Uses effective amplitudes (base + pickoff) from Trapping_Vars.
        """

        tv = self.trapVariables
        df = self.total_voltage_df
        electrodes = list(constants.electrode_names)

        # expose all dataframe columns to numexpr
        local_dict = {col: df[col] for col in df.columns}

        # DC: scalar term
        dc_map = tv.get_drive_amplitudes(tv.dc_key)  # {electrode: volts}
        v_dc_terms = []
        for el in electrodes:
            col_v = f"{el}_V"
            if col_v in df.columns:
                v_dc_terms.append(f"{col_v} * {dc_map.get(el, 0.0)}")
        sum_v_dc = " + ".join(v_dc_terms) or "0"

        ## In replacemnt testing 11/11/2025
        # ---------- choose fastest non-DC for pseudopotential ----------
        # non_dc_drives = [dk for dk in tv.get_drives() if dk.f_uHz != 0]
        # fastest = max(non_dc_drives, key=lambda dk: dk.f_uHz, default=None)

        # pseudo_expr = "0"
        # if fastest is not None:
        #     rf_map = tv.get_drive_amplitudes(fastest)  # {electrode: volts}
        #     ex_terms, ey_terms, ez_terms = [], [], []
        #     for el in electrodes:
        #         A_e = rf_map.get(el, 0.0)
        #         if A_e == 0.0:
        #             continue  # shrink the expression if this electrode doesn't contribute
        #         col_ex = f"{el}_Ex"
        #         col_ey = f"{el}_Ey"
        #         col_ez = f"{el}_Ez"
        #         if col_ex in df.columns:
        #             ex_terms.append(f"{col_ex} * {A_e}")
        #         if col_ey in df.columns:
        #             ey_terms.append(f"{col_ey} * {A_e}")
        #         if col_ez in df.columns:
        #             ez_terms.append(f"{col_ez} * {A_e}")

        #     sum_ex = " + ".join(ex_terms) or "0"
        #     sum_ey = " + ".join(ey_terms) or "0"
        #     sum_ez = " + ".join(ez_terms) or "0"

        #     Omega2 = (2.0 * math.pi * fastest.f_hz) ** 2
        #     alpha = constants.ion_charge / (
        #         4.0 * constants.ion_mass * Omega2
        #     )  # volts per (V/m)^2

        #     pseudo_expr = f"({alpha}) * (({sum_ex})**2 + ({sum_ey})**2 + ({sum_ez})**2)"

        # # final DC + pseudo
        # df["Static_TotalV"] = ne.evaluate(
        #     f"{sum_v_dc} + ({pseudo_expr})", local_dict=local_dict
        # )

        # ---------- RF pseudopotential(s) ----------
        non_dc_drives = [dk for dk in tv.get_drives() if dk.f_uHz != 0]

        pseudo_expr = "0"
        if getattr(constants, "INCLUDE_ALL_RF_PSEUDOPOTENTIALS", False):
            # Sum pseudo from every non-DC drive:  sum_i (q/(4 m Ω_i^2)) |E_i|^2
            pseudo_terms = []
            for dk in non_dc_drives:
                rf_map = tv.get_drive_amplitudes(dk)  # {electrode: volts}

                ex_terms, ey_terms, ez_terms = [], [], []
                for el in electrodes:
                    A_e = rf_map.get(el, 0.0)
                    if A_e == 0.0:
                        continue
                    cx, cy, cz = f"{el}_Ex", f"{el}_Ey", f"{el}_Ez"
                    if cx in df.columns:
                        ex_terms.append(f"{cx} * {A_e}")
                    if cy in df.columns:
                        ey_terms.append(f"{cy} * {A_e}")
                    if cz in df.columns:
                        ez_terms.append(f"{cz} * {A_e}")

                sum_ex = " + ".join(ex_terms) or "0"
                sum_ey = " + ".join(ey_terms) or "0"
                sum_ez = " + ".join(ez_terms) or "0"

                Omega2 = (2.0 * math.pi * dk.f_hz) ** 2
                alpha = constants.ion_charge / (4.0 * constants.ion_mass * Omega2)
                pseudo_terms.append(
                    f"({alpha}) * (({sum_ex})**2 + ({sum_ey})**2 + ({sum_ez})**2)"
                )

            pseudo_expr = " + ".join(pseudo_terms) or "0"
        else:
            # current fastest-only behavior (unchanged)
            fastest = max(non_dc_drives, key=lambda d: d.f_uHz, default=None)
            if fastest is not None:
                rf_map = tv.get_drive_amplitudes(fastest)
                ex_terms, ey_terms, ez_terms = [], [], []
                for el in electrodes:
                    A_e = rf_map.get(el, 0.0)
                    if A_e == 0.0:
                        continue
                    cx, cy, cz = f"{el}_Ex", f"{el}_Ey", f"{el}_Ez"
                    if cx in df.columns:
                        ex_terms.append(f"{cx} * {A_e}")
                    if cy in df.columns:
                        ey_terms.append(f"{cy} * {A_e}")
                    if cz in df.columns:
                        ez_terms.append(f"{cz} * {A_e}")

                sum_ex = " + ".join(ex_terms) or "0"
                sum_ey = " + ".join(ey_terms) or "0"
                sum_ez = " + ".join(ez_terms) or "0"

                Omega2 = (2.0 * math.pi * fastest.f_hz) ** 2
                alpha = constants.ion_charge / (4.0 * constants.ion_mass * Omega2)
                pseudo_expr = (
                    f"({alpha}) * (({sum_ex})**2 + ({sum_ey})**2 + ({sum_ez})**2)"
                )

        # final DC + pseudo (unchanged)
        df["Static_TotalV"] = ne.evaluate(
            f"{sum_v_dc} + ({pseudo_expr})", local_dict=local_dict
        )

        # ---------- per-drive totals (scalar only, no pseudo) ----------

        for dk in non_dc_drives:
            amp_map = tv.get_drive_amplitudes(dk)  # {electrode: volts}
            v_terms = []
            for el in electrodes:
                col_v = f"{el}_V"
                if col_v in df.columns:
                    coef = amp_map.get(el, 0.0)
                    if coef != 0.0:
                        v_terms.append(f"{col_v} * {coef}")
            expr = " + ".join(v_terms) or "0"
            df[drive_colname(dk)] = ne.evaluate(expr, local_dict=local_dict)

        # done
        return

    def change_electrode_variables(self, new_vars: Trapping_Vars):
        """
        Change the electrode variables of the simulation to new_vars and update the total voltage DataFrame.
        """
        self.trapVariables = new_vars
        self.update_total_voltage_columns()

    def clear_held_results(self):
        self.center_fits = {}  # Placeholder for the center fit model
        self.ion_equilibrium_positions = {}
        self.ion_eigenvectors = {}
        self.ion_eigenvalues = {}
        self.normal_modes_and_frequencies = {}
        self.driven_g_0_2_couplings = {}
        self.driven_g_0_3_couplings = {}
        self.inherent_g_0_3_couplings = {}
        self.inherent_g_0_4_couplings = {}

        self.principal_dirs = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        self.principal_dir_normalmodes_andfrequencies = {}

    def get_total_E_components(self, drive: Optional[DriveKey] = None):
        """
        Return total (Ex, Ey, Ez) across all electrodes for a given drive.
        If drive is None, uses the DC drive.
        """
        if drive is None:
            drive = self.trapVariables.dc_key

        df = self.total_voltage_df
        amp_map = self.trapVariables.get_drive_amplitudes(drive)

        ex = np.zeros(len(df), dtype=float)
        ey = np.zeros(len(df), dtype=float)
        ez = np.zeros(len(df), dtype=float)

        for el, coef in amp_map.items():
            if coef == 0.0:
                continue
            cx, cy, cz = f"{el}_Ex", f"{el}_Ey", f"{el}_Ez"
            if cx in df.columns:
                ex += df[cx].to_numpy(dtype=float) * coef
            if cy in df.columns:
                ey += df[cy].to_numpy(dtype=float) * coef
            if cz in df.columns:
                ez += df[cz].to_numpy(dtype=float) * coef

        return ex, ey, ez

    def get_total_E_magnitude(self, drive: Optional[DriveKey] = None):
        """
        Return |E| across all electrodes for a given drive.
        If drive is None, uses the DC drive.
        """
        ex, ey, ez = self.get_total_E_components(drive=drive)
        return np.sqrt(ex * ex + ey * ey + ez * ez)

    def get_yz_plane_E_magnitude(
        self,
        span_um: float = 30.0,
        x_center_m: float = 0.0,
        drive: Optional[DriveKey] = None,
    ):
        """
        Return (Y, Z, Emag, x_plane) for the yz-plane at the nearest x grid.
        span_um is total width/height of the square plane in microns.
        """
        df = self.total_voltage_df
        half = 0.5 * span_um * 1e-6

        x_vals = np.unique(df["x"].to_numpy())
        if x_vals.size == 0:
            raise ValueError("No x values found in dataframe.")
        x0 = float(x_vals[np.argmin(np.abs(x_vals - x_center_m))])

        mask = (
            (df["x"] == x0)
            & (df["y"].between(-half, half))
            & (df["z"].between(-half, half))
        )
        plane = df.loc[mask, ["x", "y", "z"]].copy()
        if plane.empty:
            raise ValueError("No points found in the requested yz plane window.")

        ex, ey, ez = self.get_total_E_components(drive=drive)
        emag = np.sqrt(ex * ex + ey * ey + ez * ez)
        plane["Emag"] = emag[mask.to_numpy()]

        y_vals = np.sort(plane["y"].unique())
        z_vals = np.sort(plane["z"].unique())
        grid = plane.pivot_table(index="y", columns="z", values="Emag")

        Y, Z = np.meshgrid(y_vals, z_vals, indexing="xy")
        E = grid.to_numpy().T

        return Y, Z, E, x0

    def get_total_MM_magnitude(self):
        """
        Return micromotion amplitude on a 30x30 um yz plane at x≈0.
        Uses the fastest non-DC drive and MM = (q / (m * f^2)) * |E|.
        """
        rf_drives = [dk for dk in self.trapVariables.get_drives() if dk.f_uHz != 0]
        if not rf_drives:
            raise ValueError("No non-DC drives found for micromotion calculation.")
        drive = max(rf_drives, key=lambda d: d.f_hz)

        Y, Z, E, x0 = self.get_yz_plane_E_magnitude(
            span_um=60.0,
            x_center_m=0.0,
            drive=drive,
        )

        omega = 2*np.pi*drive.f_hz
        const = constants.ion_charge / (constants.ion_mass * omega**2)
        print(const)
        MM = const * E
        return Y, Z, MM, x0

    def plot_total_MM_magnitude(self):
        """
        Plot micromotion amplitude on a 30x30 um yz plane at x≈0.
        """
        Y, Z, MM, x0 = self.get_total_MM_magnitude()
        fig, ax = plt.subplots(1, 1, figsize=(5.2, 4.4), constrained_layout=True)
        im = ax.contourf(Y * 1e6, Z * 1e6, MM, levels=40, cmap="viridis")
        ax.set_title(f"MM amplitude @ x={x0 * 1e6:.3g} um")
        ax.set_xlabel("y (um)")
        ax.set_ylabel("z (um)")
        fig.colorbar(im, ax=ax, label="MM amplitude (m)")
        return fig

    def _smoke_test_new_stack(self, n_ions=2, poly_deg=4):
        import numpy as np

        print("[smoke] building TotalV columns…")
        self.update_total_voltage_columns()

        # sanity: columns exist
        assert "Static_TotalV" in self.total_voltage_df.columns
        for dk in self.trapVariables.get_drives():
            if dk != self.trapVariables.dc_key:
                assert (
                    drive_colname(dk) in self.total_voltage_df.columns
                ), f"Missing {drive_colname(dk)}"

        print("[smoke] fitting center polys…")
        self.update_center_polys(polyfit_deg=poly_deg)
        assert self.center_fits.get(self.trapVariables.dc_key) is not None
        for dk in self.trapVariables.get_drives():
            if dk != self.trapVariables.dc_key:
                assert self.center_fits.get(dk) is not None

        print("[smoke] finding U minimum…")
        self.find_equilib_position_single(num_ions=n_ions)
        eq_pos = self.ion_equilibrium_positions.get(n_ions)
        # print(self.ion_equilibrium_positions.get(1))
        # print("")
        # print(self.ion_equilibrium_positions.get(2))
        # print("")
        # print(self.ion_equilibrium_positions.get(3))
        # print("")
        # print(self.ion_equilibrium_positions.get(4))

        # assert eq.shape == (n_ions, 3)

        print("[smoke] Hessian & modes…")
        H = self.get_eq_U_hessian(n_ions)
        assert H.shape == (3 * n_ions, 3 * n_ions)
        vals, vecs = self.get_mode_eigenvec_and_val(n_ions)
        assert len(vals) == 3 * n_ions and vecs.shape == (3 * n_ions, 3 * n_ions)
        omega = np.sqrt(abs(vals) / constants.ion_mass)
        f_Hz = omega / (2 * np.pi)
        f_MHz = f_Hz * 1e-6
        # print("secular frequencies (MHz):", f_MHz)  ### THIS IS THE CORRECT UNITS ###
        # print("")
        self.get_static_normal_modes_and_freq(n_ions)
        # print(self.normal_modes_and_frequencies)

        print("[smoke] 3rd/4th derivative contractions…")
        g3 = self.get_3_wise_mode_couplingss(n_ions)
        g4 = self.get_4_wise_mode_couplingss(n_ions)
        # touch a couple entries if present
        n3 = 3 * n_ions
        assert g3.shape == (n3, n3, n3), f"Unexpected shape: {g3.shape}"
        assert g4.shape == (n3, n3, n3, n3), f"Unexpected shape: {g4.shape}"

        # assert np.allclose(g3, g3.transpose(), atol=1e-10), "g3 is not symmetric"
        # assert np.allclose(
        #     g4, g4.transpose((0, 1, 3, 2)), atol=1e-10
        # ), "g4 is not symmetric"

        print("[smoke] OK")

        print("extras")
        print(" ")

    #############################OLD####################

    def update_total_voltage_OLD(self):
        """Update the total voltage DataFrame by forming
        an equation based on the electorde_varaibles and
        applying to the column "TotalV" in the dataframe.

        yes this implementation seems jank but its quick
        again this is place where errors will happen if the geometry evolves

        this is where the pseudopotential is applied.
        NOTE for now this is only taking into account the highest frequency modulation in the trappingvars
        TODO: Make the pseudopotential take into account all the modulations
        """

        drives = self.trapVariables.get_drives()
        # find the fastest drive and return the drivekey
        max_freq = 0
        max_drive = None
        for drive in drives:
            if drive.f_hz >= max_freq:
                max_freq = drive.f_hz
                max_drive = drive
        if max_drive is None:
            raise ValueError("No positive freq drives found in trapping variables.")

        evaluation_final = ""
        eval_ex_str = ""
        eval_ey_str = ""
        eval_ez_str = ""
        for electro in [
            "RF1",
            "RF2",
            "DC1",
            "DC2",
            "DC3",
            "DC4",
            "DC5",
            "DC6",
            "DC7",
            "DC8",
            "DC9",
            "DC10",
        ]:
            evaluation_final += (
                electro
                + "_V * "
                + str(self.trapVariables.get_amp(DriveKey(0, 0, "DC"), electro))
                + " + "
            )
            eval_ex_str += (
                electro
                + "_Ex * "
                + str(self.trapVariables.get_amp(max_drive, electro))
                + " + "
            )
            eval_ey_str += (
                electro
                + "_Ey * "
                + str(self.trapVariables.get_amp(max_drive, electro))
                + " + "
            )
            eval_ez_str += (
                electro
                + "_Ez * "
                + str(self.trapVariables.get_amp(max_drive, electro))
                + " + "
            )
        eval_ex_str = eval_ex_str[:-3]  # remove last " + "
        eval_ey_str = eval_ey_str[:-3]  # remove last " + "
        eval_ez_str = eval_ez_str[:-3]  # remove last " + "

        evaluation_pseudo_part = (
            "("
            + str(constants.ion_charge)
            + " * (("
            + eval_ex_str
            + ")**2 + ("
            + eval_ey_str
            + ")**2 + ("
            + eval_ez_str
            + ")**2) / (4 * "
            + str(constants.ion_mass)
            + " * "
            + str(max_freq**2)
            + "))"
        )

        evaluation_final += evaluation_pseudo_part

        # print("evaluation_final: ", evaluation_final)

        self.total_voltage_df["TotalV"] = ne.evaluate(
            evaluation_final,
            local_dict={
                "RF1_V": self.total_voltage_df["RF1_V"],
                "RF2_V": self.total_voltage_df["RF2_V"],
                "DC1_V": self.total_voltage_df["DC1_V"],
                "DC2_V": self.total_voltage_df["DC2_V"],
                "DC3_V": self.total_voltage_df["DC3_V"],
                "DC4_V": self.total_voltage_df["DC4_V"],
                "DC5_V": self.total_voltage_df["DC5_V"],
                "DC6_V": self.total_voltage_df["DC6_V"],
                "DC7_V": self.total_voltage_df["DC7_V"],
                "DC8_V": self.total_voltage_df["DC8_V"],
                "DC9_V": self.total_voltage_df["DC9_V"],
                "DC10_V": self.total_voltage_df["DC10_V"],
                "RF1_Ex": self.total_voltage_df["RF1_Ex"],
                "RF2_Ex": self.total_voltage_df["RF2_Ex"],
                "DC1_Ex": self.total_voltage_df["DC1_Ex"],
                "DC2_Ex": self.total_voltage_df["DC2_Ex"],
                "DC3_Ex": self.total_voltage_df["DC3_Ex"],
                "DC4_Ex": self.total_voltage_df["DC4_Ex"],
                "DC5_Ex": self.total_voltage_df["DC5_Ex"],
                "DC6_Ex": self.total_voltage_df["DC6_Ex"],
                "DC7_Ex": self.total_voltage_df["DC7_Ex"],
                "DC8_Ex": self.total_voltage_df["DC8_Ex"],
                "DC9_Ex": self.total_voltage_df["DC9_Ex"],
                "DC10_Ex": self.total_voltage_df["DC10_Ex"],
                "RF1_Ey": self.total_voltage_df["RF1_Ey"],
                "RF2_Ey": self.total_voltage_df["RF2_Ey"],
                "DC1_Ey": self.total_voltage_df["DC1_Ey"],
                "DC2_Ey": self.total_voltage_df["DC2_Ey"],
                "DC3_Ey": self.total_voltage_df["DC3_Ey"],
                "DC4_Ey": self.total_voltage_df["DC4_Ey"],
                "DC5_Ey": self.total_voltage_df["DC5_Ey"],
                "DC6_Ey": self.total_voltage_df["DC6_Ey"],
                "DC7_Ey": self.total_voltage_df["DC7_Ey"],
                "DC8_Ey": self.total_voltage_df["DC8_Ey"],
                "DC9_Ey": self.total_voltage_df["DC9_Ey"],
                "DC10_Ey": self.total_voltage_df["DC10_Ey"],
                "RF1_Ez": self.total_voltage_df["RF1_Ez"],
                "RF2_Ez": self.total_voltage_df["RF2_Ez"],
                "DC1_Ez": self.total_voltage_df["DC1_Ez"],
                "DC2_Ez": self.total_voltage_df["DC2_Ez"],
                "DC3_Ez": self.total_voltage_df["DC3_Ez"],
                "DC4_Ez": self.total_voltage_df["DC4_Ez"],
                "DC5_Ez": self.total_voltage_df["DC5_Ez"],
                "DC6_Ez": self.total_voltage_df["DC6_Ez"],
                "DC7_Ez": self.total_voltage_df["DC7_Ez"],
                "DC8_Ez": self.total_voltage_df["DC8_Ez"],
                "DC9_Ez": self.total_voltage_df["DC9_Ez"],
                "DC10_Ez": self.total_voltage_df["DC10_Ez"],
            },
        )

        return

    # Important and old
    def get_voltage_poly_at_center_old(
        self, lookaround_x=20, lookaround_yz=5, polyfit=4, max_pnts=1e6
    ):
        return self.get_voltage_poly_at_region_old(
            x_low=-lookaround_x,
            x_high=lookaround_x,
            y_low=-lookaround_yz,
            y_high=lookaround_yz,
            z_low=-lookaround_yz,
            z_high=lookaround_yz,
            max_pnts=max_pnts,
            polyfit=polyfit,
        )

    # Important and old
    def get_voltage_poly_at_region_old(
        self,
        x_low=-10,
        x_high=10,
        y_low=-10,
        y_high=10,
        z_low=-10,
        z_high=10,
        max_pnts=1e6,
        polyfit=4,
    ):

        # get initial maxcutout
        cutout_of_df = self.total_voltage_df[
            (self.total_voltage_df["x"].between((x_low * 1e-6), (x_high * 1e-6)))
            & (self.total_voltage_df["y"].between(y_low * 1e-6, y_high * 1e-6))
            & (self.total_voltage_df["z"].between(z_low * 1e-6, z_high * 1e-6))
        ]

        # if len(cutout_of_df) > max_pnts then randomply sample max_pnts points
        if len(cutout_of_df) > max_pnts:
            cutout_of_df = cutout_of_df.sample(n=max_pnts, random_state=1)
        print("len(cutout_of_df): ", len(cutout_of_df))

        voltage_vals = cutout_of_df["TotalV"].values
        print(len(cutout_of_df), " points found in cutout")
        # print("voltage_vals: ", voltage_vals)
        xyz_vals = cutout_of_df[["x", "y", "z"]].values

        poly = PolynomialFeatures(degree=polyfit, include_bias=True)
        X_poly = poly.fit_transform(xyz_vals)

        # Fit the model
        model = LinearRegression()
        model.fit(X_poly, voltage_vals)

        # find and print out the r2 of the fit
        r2 = model.score(X_poly, voltage_vals)
        if r2 < 0.999:
            print(
                "Warning: Low R-squared value for polynomial fit done by get_voltage_poly_at_region"
            )
            print("R-squared of the fit:", r2)

        # return the polynomial
        return model, poly, r2

    # Single Ion Analysis Old
    def get_principal_freq_at_min_old(
        self, getall=False, fitdeg=4, look_around=5, return_coefs=False
    ):
        """
        This fuction finds the principal frequency at the minimum point of the total voltage.
        To do so the minimum in the potential well is found and then at this point a R3 fit is performed to find the eigenfrequencies and eigenvectors.

        args:
            getmintoo (bool): If True, return the minimum points as well. Default is False.
            fitdeg (int): The degree of the polynomial to fit to the data. Default is 4.
            look_around (float): The size of the cutout around the minimum point to use for the R3 fit. Default is 5 microns.

        returns:
            eigenfreq (list): A list of the eigenfrequencies in increasing order.
            eigendir (dict): A dictionary of the eigenvectors in a readable format.
            min1 (array): The best fit minimum point (x,y,z) in meters.
            min2 (array): The original minimum point (x,y,z) in meters.
        """
        min1, min2 = self.find_V_min()
        if return_coefs or getall:
            eigenfreq, axialfreq, eigendir, coefs = self.get_freqs_at_point_withR3_fit(
                min1[0],
                min1[1],
                min1[2],
                look_around=look_around,
                polyfit=fitdeg,
                return_coefs=True,
            )
        else:
            eigenfreq, axialfreq, eigendir = self.get_freqs_at_point_withR3_fit(
                min1[0],
                min1[1],
                min1[2],
                look_around=look_around,
                polyfit=fitdeg,
                return_coefs=False,
            )

        # Convert eigenfreq to a numpy array for sorting
        eigenfreq = np.array(eigenfreq)

        # Sort eigenfreq in increasing order and eigendir accordingly
        sorted_indices = np.argsort(eigenfreq)
        eigenfreq = eigenfreq[sorted_indices].tolist()  # Convert back to list if needed
        eigendir = eigendir[
            :, sorted_indices
        ]  # Sort eigendir columns based on sorted indices

        # Convert eigenvectors into a readable format
        eigendir_readable = {
            f"Direction {i+1}": f"({vec[0]:.3f}, {vec[1]:.3f}, {vec[2]:.3f})"
            for i, vec in enumerate(
                eigendir.T
            )  # Transpose so each column is an eigenvector
        }
        if getall:
            return eigenfreq, eigendir_readable, min1, min2, coefs

        return eigenfreq, eigendir_readable

    # * * * * * * * * * * * ** * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *#
    # Unknown (old?)
    def get_U_min(self, number_of_ions: int):
        """
        Minimizes the total potential energy (trap potential + Coulomb)
        for 'number_of_ions' ions.
        """

        def U_func(positions_flat):
            n = len(positions_flat) // 3
            positions = positions_flat.reshape((n, 3))

            U_trap = 0.0
            U_coulomb = 0.0

            for i in range(n):
                x, y, z = positions[i]
                U_trap += self.find_V_trap_at_point(x, y, z) * constants.ion_charge

            epsilon = 1e-12  # small buffer to avoid division-by-zero issues
            for i in range(n):
                for j in range(i + 1, n):
                    dist = np.linalg.norm(positions[i] - positions[j])
                    dist = max(dist, epsilon)
                    U_coulomb += (
                        constants.coulomb_constant * constants.ion_charge**2
                    ) / dist

            total_energy = (
                U_trap + U_coulomb
            ) * 1e25  # scaling for numerical stability
            return total_energy

        def grad_U_func(positions_flat):
            n = len(positions_flat) // 3
            positions = positions_flat.reshape((n, 3))
            grad = np.zeros_like(positions)

            # Coulomb term
            for i in range(n):
                for j in range(i + 1, n):
                    r_i = positions[i]
                    r_j = positions[j]
                    diff = r_i - r_j
                    dist = np.linalg.norm(diff) + 1e-12  # to prevent division by 0

                    force = (
                        (constants.coulomb_constant * constants.ion_charge**2)
                        * diff
                        / dist**3
                    )

                    grad[i] += force  # ∂U/∂r_i
                    grad[j] -= force  # ∂U/∂r_j, opposite direction

            # Trap term (assuming you can get ∂V/∂x, ∂V/∂y, ∂V/∂z)
            for i in range(n):
                x, y, z = positions[i]

                # estimate gradient of trap potential
                dx = 1e-7  # small finite difference
                # dV_dx = (self.find_V_trap_at_point(x + dx*30, y, z) - self.find_V_trap_at_point(x - dx*30, y, z)) / (2 * dx*30)
                # dV_dy = (self.find_V_trap_at_point(x, y + dx, z) - self.find_V_trap_at_point(x, y - dx, z)) / (2 * dx)
                # dV_dz = (self.find_V_trap_at_point(x, y, z + dx) - self.find_V_trap_at_point(x, y, z - dx)) / (2 * dx)
                # print(np.array([dV_dx, dV_dy, dV_dz]))

                # use the reg gucntion
                dV_dx, dV_dy, dV_dz = self.find_V_trap_at_point(x, y, z, derivs=True)

                grad[i] += constants.ion_charge * np.array([dV_dx, dV_dy, dV_dz])

            return grad.flatten() * 1e25  # match scaling of U_func

        init_guess_flat = np.array(
            constants.ion_locations_intial_guess[number_of_ions]
        ).flatten()

        bounds = constants.ion_locations_bounds[number_of_ions]

        result = minimize(
            U_func,
            init_guess_flat,
            method="L-BFGS-B",
            jac=grad_U_func,  # <-- pass the gradient
            bounds=bounds,
            options={"eps": 1e-6, "disp": False, "maxiter": 10000, "maxfun": 100000},
        )

        if not result.success:
            print("Minimization failed:", result.message)
        else:
            print("Minimization successful!")
            print("Final potential energy:", result.fun / 1e20)

        # return just the minimized positions
        return result.x.reshape((number_of_ions, 3))

    # Unknown (old?)
    def get_U_min_new(self, number_of_ions: int):
        """
        Minimizes the total potential energy (trap + Coulomb) for 'number_of_ions' ions
        using:
        1) coordinate rescaling in y,z to reduce stiffness,
        2) offsetting the potential so the initial guess is U=0,
        3) basinhopping for global search, with L-BFGS-B for local minimization.
        """

        # --
        # 1. Choose a rescaling factor for y, z
        #    This helps the optimizer handle extremely stiff Y/Z directions.
        scale_yz = 1e-1

        def scale_positions(x_flat):
            """Take an array of shape (3*n) and scale y,z coords."""
            x = x_flat.copy().reshape((-1, 3))
            x[:, 1] *= scale_yz
            x[:, 2] *= scale_yz
            return x

        def unscale_positions(x_flat_scaled):
            """Inverse of scale_positions."""
            x = x_flat_scaled.copy().reshape((-1, 3))
            x[:, 1] /= scale_yz
            x[:, 2] /= scale_yz
            return x

        # ---------------------------------------------------
        # 2. Define the total potential in unscaled space
        #    We'll shift it so that the initial guess is at U=0.
        def total_potential_unscaled(positions_flat):
            n = len(positions_flat) // 3
            positions = positions_flat.reshape((n, 3))

            U_trap = 0.0
            U_coulomb = 0.0

            # Trap potential
            for i in range(n):
                x, y, z = positions[i]
                U_trap += self.find_V_trap_at_point(x, y, z) * constants.ion_charge

            # Coulomb potential
            epsilon = 1e-12
            for i in range(n):
                for j in range(i + 1, n):
                    dist = np.linalg.norm(positions[i] - positions[j])
                    dist = max(dist, epsilon)
                    U_coulomb += (
                        constants.coulomb_constant * constants.ion_charge**2
                    ) / dist

            return (U_trap + U_coulomb) * 1e25  # scale for numeric stability

        # ---------------------------------------------------
        # 3. Build the scaled + offset version of the potential
        #    We'll store the potential at the initial guess, subtract it so at guess => 0.
        init_guess = np.array(constants.ion_locations_intial_guess[number_of_ions])
        init_guess_flat = init_guess.flatten()
        init_guess_scaled_flat = scale_positions(init_guess_flat).ravel()

        U0 = total_potential_unscaled(init_guess_flat)  # offset

        def U_func_scaled(x_flat_scaled):
            # Unscale back to physical coords
            x_flat = unscale_positions(x_flat_scaled).ravel()
            return total_potential_unscaled(x_flat) - U0  # shift so init guess => 0

        def grad_U_func_scaled(x_flat_scaled):
            """
            Analytical gradient w.r.t. scaled coords
            = dU/dx_scaled. We'll do chain rule:
            dU/dx_scaled = dU/dx_unscaled * dx_unscaled/dx_scaled.
            """
            x_flat = unscale_positions(x_flat_scaled).ravel()
            n = len(x_flat) // 3
            positions = x_flat.reshape((n, 3))
            grad_unscaled = np.zeros_like(positions)

            # Coulomb gradient
            for i in range(n):
                for j in range(i + 1, n):
                    diff = positions[i] - positions[j]
                    dist = np.linalg.norm(diff) + 1e-12
                    force = (
                        (constants.coulomb_constant * constants.ion_charge**2)
                        * diff
                        / dist**3
                    )
                    grad_unscaled[i] += force
                    grad_unscaled[j] -= force

            # Trap gradient
            # Use your direct derivative approach if available
            # (here let's just do small finite differences on trap, or use your find_V_trap_at_point(derivs=True))
            dx = 1e-7
            for i in range(n):
                x, y, z = positions[i]
                # If you have a direct derivative method, call it. E.g.:
                dV_dx, dV_dy, dV_dz = self.find_V_trap_at_point(x, y, z, derivs=True)

                grad_unscaled[i] += constants.ion_charge * np.array(
                    [dV_dx, dV_dy, dV_dz]
                )

            # Scale by 1e25 to match the total potential scaling
            grad_unscaled *= 1e25

            # Now correct for scaling of y,z
            # x_scaled = [x, y*scale_yz, z*scale_yz]
            # => dx/dx_scaled = 1 for x
            # => dy/dy_scaled = scale_yz
            # => dz/dz_scaled = scale_yz
            # So dU/dx_scaled[0] = dU/dx_unscaled[0]
            #    dU/dx_scaled[1] = dU/dx_unscaled[1] * scale_yz
            # etc.
            grad_unscaled[:, 1] *= scale_yz
            grad_unscaled[:, 2] *= scale_yz

            return grad_unscaled.ravel()

        # ---------------------------------------------------
        # 4. Define BOUNDS in scaled space
        raw_bounds = constants.ion_locations_bounds[number_of_ions]
        scaled_bounds = []
        for (xmin, xmax), (ymin, ymax), (zmin, zmax) in raw_bounds:
            scaled_bounds.append((xmin, xmax))  # x
            scaled_bounds.append((ymin * scale_yz, ymax * scale_yz))  # y
            scaled_bounds.append((zmin * scale_yz, zmax * scale_yz))  # z

        # ---------------------------------------------------
        # 5. Set up local minimizer (L-BFGS-B)
        minimizer_kwargs = {
            "method": "L-BFGS-B",
            "jac": grad_U_func_scaled,
            "bounds": scaled_bounds,
            "options": {"disp": False, "maxiter": 10000, "maxfun": 100000, "eps": 1e-7},
        }

        # ---------------------------------------------------
        # 6. Basinhopping (global + local approach)
        #    We'll do multiple basins to try to find global minimum.
        #    You can tweak niter for how many random steps to take.
        niter = 10

        # We'll define a trivial "take_step" or just use the default.
        # The default random step size might be a bit large, so let's define our own:
        class ScaledRandomStep(object):
            def __init__(self, step_size=1e-7):
                self.step_size = step_size

            def __call__(self, x):
                # random step in scaled coords
                return x + np.random.uniform(
                    -self.step_size, self.step_size, size=x.shape
                )

        bh = basinhopping(
            U_func_scaled,
            init_guess_scaled_flat,
            minimizer_kwargs=minimizer_kwargs,
            niter=niter,
            stepsize=5e-7,  # or pass in a custom "take_step"
            disp=True,
        )

        if not bh.lowest_optimization_result.success:
            print(
                "Minimization (basinhopping) failed:",
                bh.lowest_optimization_result.message,
            )
        else:
            print("Minimization successful!")
            print(
                "Final potential energy (shifted) =", bh.lowest_optimization_result.fun
            )
            # unshifted = ...
            # If you want the absolute potential, just add U0 back:
            # absolute_final_energy = bh.lowest_optimization_result.fun + U0

        # ---------------------------------------------------
        # 7. Unscale the final best positions for physical coordinates
        best_x_scaled = bh.lowest_optimization_result.x
        best_positions = unscale_positions(best_x_scaled)

        print("Final positions (m):\n", best_positions)

        return best_positions

    # Unknown (old?)
    def get_smooth_Ufunc_and_UjaxFunc(self, number_of_ions: int, elec_v_model=None):
        """ """
        if elec_v_model is None:
            elec_v_model = self.get_voltage_poly_at_center_old(polyfit=4, max_pnts=1e6)[
                0
            ]

        def U_func(positions_flat):
            n = len(positions_flat) // 3
            positions = positions_flat.reshape((n, 3))

            U_trap = 0.0
            U_coulomb = 0.0

            for i in range(n):
                x, y, z = positions[i]
                U_trap += self.find_V_trap_at_point(x, y, z) * constants.ion_charge

            epsilon = 1e-12  # small buffer to avoid division-by-zero issues
            for i in range(n):
                for j in range(i + 1, n):
                    dist = np.linalg.norm(positions[i] - positions[j])
                    dist = max(dist, epsilon)
                    U_coulomb += (
                        constants.coulomb_constant * constants.ion_charge**2
                    ) / dist

            total_energy = (
                U_trap + U_coulomb
            ) * 1e25  # scaling for numerical stability
            return total_energy

        return U_func

    # Unknown (old?)
    def get_U_min_using_polyfit(self, number_of_ions: int):
        """
        This fucnction will minimize the total_U fucntion returned by polu
        """
        pass

    # * * * * * * * * * * * ** * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *#

    # Not writen
    def get_ion_mins(self):
        return


def check_E_units_on_center_line(df, electrode="RF1"):
    import numpy as np, pandas as pd

    vcol = f"{electrode}_V"
    ecol = f"{electrode}_Ex"
    for c in ("x", "y", "z", vcol, ecol):
        if c not in df.columns:
            print(f"Missing column: {c}")
            return

    # pick the y=const, z=const plane closest to (0,0)
    ys = np.unique(df["y"].to_numpy())
    zs = np.unique(df["z"].to_numpy())
    y0 = ys[np.argmin(np.abs(ys))]
    z0 = zs[np.argmin(np.abs(zs))]

    line = df[(df["y"] == y0) & (df["z"] == z0)][["x", vcol, ecol]].sort_values("x")
    line = line.drop_duplicates(subset="x")

    if len(line) < 5:
        print("Not enough points on the center line to check E units.")
        return

    x = line["x"].to_numpy()
    V = line[vcol].to_numpy()
    Ex = line[ecol].to_numpy()

    # robust slope at center via linear fit (less noisy than raw gradient)
    A = np.c_[np.ones_like(x), x]
    beta, *_ = np.linalg.lstsq(A, V, rcond=None)  # V ≈ beta0 + beta1 * x
    dVdx_fit = beta[1]

    # compare signs/magnitudes
    ratio = np.median(-Ex / dVdx_fit)
    print(f"Center line: y={y0:.3e} m, z={z0:.3e} m, points={len(line)}")
    print(
        f"median(-Ex / dVdx_fit) = {ratio:.3g}   (≈1 if Ex in V/m; ≈1e6 if Ex in V/µm)"
    )


# Idea run for each rf freq/amp pairing a search for the endcaps that gives distance between three ions to be 5um and then report the g_0 of modes 1,2 and modes 1,2 frequencyies


def main_2():
    """
    Simple single-ion run for the 2D trap dataset.
    Prints the trap secular frequencies and principal directions.
    """
    tv = Trapping_Vars()

    # RF drive setup
    rf_freq_hz = (43) * 10**6
    rf_amp_rf1 = 515
    rf_amp_rf2 = 515
    tv.add_driving("RF", rf_freq_hz, 0.0, {"RF1": rf_amp_rf1, "RF2": rf_amp_rf2})
    x = 14
    outer = 24 - x
    inner = 13 - x
    # DC electrode offsets (v)
    dc_offsets = {
        "DC1": outer,
        "DC2": inner,
        "DC3": outer,
        "DC4": outer,
        "DC5": inner,
        "DC6": outer,
        "DC7": outer,
        "DC8": inner,
        "DC9": outer,
        "DC10": outer,
        "DC11": inner,
        "DC12": outer,
        "RF1": -10,
        "RF2": -10,
    }

    # dc_offsets = {
    #     "DC1": 2.0,
    #     "DC2": 1.0,
    #     "DC3": 0.0,
    #     "DC4": 1.0,
    #     "DC5": 2.0,
    #     "DC6": 2.0,
    #     "DC7": 1.0,
    #     "DC8": 0.0,
    #     "DC9": 1.0,
    #     "DC10": 2.0,
    # }

    for el, volts in dc_offsets.items():
        tv.set_amp(tv.dc_key, el, volts)

    sim = Simulation("InnTrapFine", tv)

    # Build fits, equilibrium, and single-ion modes
    sim._smoke_test_new_stack(n_ions=1, poly_deg=4)
    sim.compute_principal_directions_from_one_ion()
    sim.populate_normalmodes_in_prinipledir_freq_labels()

    ppack = sim.principal_dir_normalmodes_andfrequencies.get(1)
    if ppack is None:
        raise RuntimeError("Principal-direction data not populated for n=1.")

    freqs_hz = np.asarray(ppack.get("frequencies_Hz"), dtype=float)
    principal_dirs = np.asarray(sim.principal_dirs, dtype=float)

    print("Single-ion trap secular frequencies (Hz):", freqs_hz)
    print("Principal directions (rows: dir_0..2 in lab x,y,z):")
    print(principal_dirs)

    # Print center-region polynomial fit for Static_TotalV (DC drive)
    center_fit = sim.center_fits.get(sim.trapVariables.dc_key)
    if center_fit is None:
        print("No center fit found for Static_TotalV.")
        return
    model, poly, r2 = center_fit
    print("Center-region Static_TotalV polynomial fit (degree=4)")
    print("R^2:", r2)
    print(
        "Fit region (um): "
        f"x=[{-constants.center_region_x_um}, {constants.center_region_x_um}], "
        f"y=[{-constants.center_region_y_um}, {constants.center_region_y_um}], "
        f"z=[{-constants.center_region_z_um}, {constants.center_region_z_um}]"
    )

    # # Plot plane cuts of the fitted polynomial at the center
    # span_x = constants.center_region_x_um * 1e-6
    # span_y = constants.center_region_y_um * 1e-6
    # span_z = constants.center_region_z_um * 1e-6
    # n = 120

    # def _eval_grid(xg, yg, zg):
    #     pts = np.c_[xg.ravel(), yg.ravel(), zg.ravel()]
    #     vals = model.predict(poly.transform(pts))
    #     return vals.reshape(xg.shape)

    # x = np.linspace(-span_x, span_x, n)
    # y = np.linspace(-span_y, span_y, n)
    # z = np.linspace(-span_z, span_z, n)

    # X_xy, Y_xy = np.meshgrid(x, y, indexing="xy")
    # Z0 = np.zeros_like(X_xy)
    # V_xy = _eval_grid(X_xy, Y_xy, Z0)

    # X_xz, Z_xz = np.meshgrid(x, z, indexing="xy")
    # Y0 = np.zeros_like(X_xz)
    # V_xz = _eval_grid(X_xz, Y0, Z_xz)

    # Y_yz, Z_yz = np.meshgrid(y, z, indexing="xy")
    # X0 = np.zeros_like(Y_yz)
    # V_yz = _eval_grid(X0, Y_yz, Z_yz)

    # fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
    # im0 = axes[0].contourf(
    #     X_xy * 1e6, Y_xy * 1e6, V_xy, levels=40, cmap="viridis"
    # )
    # axes[0].set_title("Static_TotalV fit @ z=0")
    # axes[0].set_xlabel("x (um)")
    # axes[0].set_ylabel("y (um)")
    # fig.colorbar(im0, ax=axes[0])

    # im1 = axes[1].contourf(
    #     X_xz * 1e6, Z_xz * 1e6, V_xz, levels=40, cmap="viridis"
    # )
    # axes[1].set_title("Static_TotalV fit @ y=0")
    # axes[1].set_xlabel("x (um)")
    # axes[1].set_ylabel("z (um)")
    # fig.colorbar(im1, ax=axes[1])

    # im2 = axes[2].contourf(
    #     Y_yz * 1e6, Z_yz * 1e6, V_yz, levels=40, cmap="viridis"
    # )
    # axes[2].set_title("Static_TotalV fit @ x=0")
    # axes[2].set_xlabel("y (um)")
    # axes[2].set_ylabel("z (um)")
    # fig.colorbar(im2, ax=axes[2])

    # plt.show()

    # Print second derivatives at the origin (center) for Static_TotalV fit
    d2x, d2y, d2z = sim.evaluate_center_poly_2ndderivatives(0.0, 0.0, 0.0)
    print("Second derivatives at center (V/m^2):")
    print("d2V/dx2:", d2x)
    print("d2V/dy2:", d2y)
    print("d2V/dz2:", d2z)

    # sim.plot_total_voltage_along_axis("x", 100)
    # sim.plot_total_voltage_along_axis("y", 100)
    # sim.plot_total_voltage_along_axis("z", 100)

    eq_pos = sim.ion_equilibrium_positions.get(1)
    print("Single-ion equilibrium position (m):", eq_pos)
    
    fig1 = sim.plot_total_voltage_plane_cuts()
    fig = sim.plot_total_voltage_along_axes()
    plt.show()

    sim.plot_total_MM_magnitude()

    Y, Z, E, x0 = sim.get_yz_plane_E_magnitude(
        span_um=30.0,
        x_center_m=0.0,
        drive=sim.trapVariables.dc_key,
    )

    fig, ax = plt.subplots(1, 1, figsize=(5.2, 4.4), constrained_layout=True)
    im = ax.contourf(Y * 1e6, Z * 1e6, E, levels=40, cmap="viridis")
    ax.set_title(f"|E| @ x={x0 * 1e6:.3g} um (DC drive)")
    ax.set_xlabel("y (um)")
    ax.set_ylabel("z (um)")
    fig.colorbar(im, ax=ax, label="|E| (V/m)")
    plt.show()


def main_3():
    """
    Plot |E| on the yz plane (x≈0) over a 30x30 um window.
    """
    tv = Trapping_Vars()

    # RF drive setup
    rf_freq_hz = (43) * 10**6
    rf_amp_rf1 = 515
    rf_amp_rf2 = 515
    tv.add_driving("RF", rf_freq_hz, 0.0, {"RF1": rf_amp_rf1, "RF2": rf_amp_rf2})
    outer = 0
    inner = 0
    # DC electrode offsets (v)
    dc_offsets = {
        "DC1": outer,
        "DC2": inner,
        "DC3": outer,
        "DC4": outer,
        "DC5": inner,
        "DC6": outer,
        "DC7": outer,
        "DC8": inner,
        "DC9": outer,
        "DC10": outer,
        "DC11": inner,
        "DC12": outer,
        "RF1": 1,
        "RF2": 1,
    }

    for el, volts in dc_offsets.items():
        tv.set_amp(tv.dc_key, el, volts)

    sim = Simulation("InnTrapFine", tv)

    Y, Z, E, x0 = sim.get_yz_plane_E_magnitude(
        span_um=30.0,
        x_center_m=0.0,
        drive=sim.trapVariables.dc_key,
    )

    fig, ax = plt.subplots(1, 1, figsize=(5.2, 4.4), constrained_layout=True)
    im = ax.contourf(Y * 1e6, Z * 1e6, E, levels=40, cmap="viridis")
    ax.set_title(f"|E| @ x={x0 * 1e6:.3g} um (DC drive)")
    ax.set_xlabel("y (um)")
    ax.set_ylabel("z (um)")
    fig.colorbar(im, ax=ax, label="|E| (V/m)")
    plt.show()


def main_1():
    timestart = time.time()

    tv = Trapping_Vars()
    rf = tv.add_driving("RF", 25500000, 0.0, {"RF1": 377.0, "RF2": 377.0})
    tv.apply_dc_twist_endcaps(twist=3.275, endcaps=5)  # volts
    
    
    
    tv.set_amp(tv.dc_key, "RF16", 0)

    extradrive = tv.add_driving(
        "ExtraDrive1",
        1688700,
        0.0,
        {
            "DC1": 0.175,
            "DC2": 0.060,
            "DC3": 0.0,
            "DC4": -0.60,
            "DC5": -0.175,
            "DC10": 0.175,
            "DC9": 0.060,
            "DC8": 0.0,
            "DC7": -0.60,
            "DC6": -0.175,
            "RF16": 0.1,
        },
    )

    # test_sim = Simulation("Hyper_2", tv)
    # test_sim = Simulation("NISTMock", tv)
    # test_sim = Simulation("Simp58_101", tv)
    test_sim = Simulation("twodTrap_1", tv)

    numionss = 8
    
    test_sim._smoke_test_new_stack(n_ions=numionss, poly_deg=4)

    # print(test_sim.ion_equilibrium_positions.get(numionss))
    test_sim.get_mode_eigenvec_and_val(numionss)
    norm_mode_freq = test_sim.get_static_normal_modes_and_freq(numionss)
    # print(test_sim.normal_modes_and_frequencies.get(numionss))

    g0 = test_sim.get_g0_matrix(numionss, extradrive)
    g0_3 = test_sim.get_driven_g_0_3_tensor(numionss, extradrive)
    print(" ")
    # print(g0)
    print(" ")
    # print(test_sim.find_largest_g0(numionss, extradrive))

    # print 1,2 g_0 and the corosponding mode vectors and frequencyies
    # print(test_sim.normal_modes_and_frequencies.get(numionss))
    eq_pos = test_sim.ion_equilibrium_positions.get(numionss)

    print(eq_pos)
    print(g0[1][2])
    print(g0[2][1])

    print(" hi hi hi hi hi hi hi ")
    test_sim.get_3_wise_mode_couplingss(numionss)
    test_sim.get_4_wise_mode_couplingss(numionss)

    # print(test_sim.inherent_g_0_3_couplings)
    print("")
    print("hhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh")
    print("")
    # print(test_sim.inherent_g_0_4_couplings)

    np.set_printoptions(suppress=True, linewidth=200, threshold=10**9)

    def _as_array(x):
        import numpy as np

        if isinstance(x, np.ndarray):
            return x
        if isinstance(x, dict):
            print("hi")
            # common wrappers
            for key in ("g3_Hz", "g4_Hz", "g3_rad_s", "g4_rad_s"):
                if key in x and isinstance(x[key], np.ndarray):
                    return x[key]
            # first ndarray value
            for v in x.values():
                if isinstance(v, np.ndarray):
                    return v
        return np.asarray(x)

    def _stats(name, A):
        import numpy as np

        A = np.ravel(np.abs(_as_array(A))).astype(float)
        A = A[np.isfinite(A)]
        if A.size == 0:
            print(f"{name}: no finite data")
            return

        mean = float(np.mean(A))
        median = float(np.median(A))
        q1 = float(np.percentile(A, 25))
        q3 = float(np.percentile(A, 75))
        std = float(np.std(A, ddof=0))

        # magnitude buckets (>=1): [1,10), [10,100), [100,1e3), [1e3,1e4), [1e4, +inf)
        b_n2 = int(np.sum(A < 0.01))
        b_n1 = int(np.sum((A >= 0.01) & (A < 0.1)))
        b_0 = int(np.sum((A >= 0.1) & (A < 1)))
        b1 = int(np.sum((A >= 1) & (A < 10)))
        b2 = int(np.sum((A >= 10) & (A < 100)))
        b3 = int(np.sum((A >= 100) & (A < 1000)))
        b4 = int(np.sum((A >= 1000) & (A < 10000)))
        b5 = int(np.sum(A >= 10000))

        print(
            f"{name}: count={A.size}, mean={mean:.3e}, median={median:.3e}, "
            f"Q1={q1:.3e}, Q3={q3:.3e}, std={std:.3e}"
        )
        print(
            f"  magnitude partitions: (inf, .01): {b_n2} [.01, .1): {b_n1} [.1, 1): {b_0} [1,10): {b1}, [10,100): {b2}, [100,1e3): {b3}, "
            f"[1e3,1e4): {b4}, [1e4,+): {b5}, "
        )

    # 3-wise
    G3 = _as_array(test_sim.inherent_g_0_3_couplings[numionss])
    print("\n=== Full 3-wise g0 tensor (Hz) ===")
    print(f"shape={getattr(G3, 'shape', None)}, dtype={getattr(G3, 'dtype', None)}")
    # print(G3)
    _stats("g0 (3-wise) |Hz|", G3)

    # 4-wise
    G4 = _as_array(test_sim.inherent_g_0_4_couplings[numionss])
    print("\n=== Full 4-wise g0 tensor (Hz) ===")
    print(f"shape={getattr(G4, 'shape', None)}, dtype={getattr(G4, 'dtype', None)}")
    # print(G4)
    _stats("g0 (4-wise) |Hz|", G4)

    # driven 2-wise
    DG2 = _as_array(test_sim.driven_g_0_2_couplings[extradrive][numionss])
    print("\n=== Full 4-wise g0 tensor (Hz) ===")
    print(f"shape={getattr(DG2, 'shape', None)}, dtype={getattr(DG2, 'dtype', None)}")
    # print(DG2)

    _stats("d0 (2-wise) |Hz|", DG2)
    print("hi")
    _stats("g0 (3-wise) |Hz|", G3)
    print("hi")
    _stats("g0 (4-wise) |Hz|", G4)

    print("")
    print(test_sim.ion_equilibrium_positions.get(numionss))

    timeend = time.time()
    print(timeend - timestart)

    out = test_sim.collect_resonant_couplings(num_ions=numionss, tol_Hz=1e3)
    import pprint

    print("\n=== Full 'out' (pprint) ===")
    pprint.pprint(out, width=120, compact=False)

    print(" normalmodes: ")
    print(" ")
    print(norm_mode_freq)
    print(" ")
    print(eq_pos)
    print(" ")
    print(g0_3)
    print(" ")
    print(g0)
    
    time11 = time.time()
    print(test_sim.compute_principal_directions_from_one_ion())
    test_sim.populate_normalmodes_in_prinipledir_freq_labels()
    time12 = time.time()
    print(" ")
    print("stuff")
    print(" ")
    print(test_sim.principal_dir_normalmodes_andfrequencies)
    time13 = time.time()
    print(" Time for principal directions: ", time12 - time11)
    print(" Time for principledir: ", time13 - time12)
    print(test_sim.ion_equilibrium_positions.get(numionss))

    eq_pos = test_sim.ion_equilibrium_positions.get(numionss)
    if eq_pos is not None:
        eq_pos = np.asarray(eq_pos, dtype=float)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(eq_pos[:, 0], eq_pos[:, 1], eq_pos[:, 2], s=30)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_zlabel("z (m)")
        ax.set_title(f"Equilibrium positions (n={numionss})")
        plt.tight_layout()
        plt.show()
    else:
        print("No equilibrium positions found to plot.")

    print(" End of main ")


if __name__ == "__main__":
    main_2()

# find and print the avg, median, uper lower quartile mean std of the g_0 couplings for 3 and 4 wise couplings


# print("hi")
# t1 = time.time()


# evaribs = evars.get_electrodvars_w_twist(377, 25500000 * 2 * math.pi, -0.275, 5)
# evaribs.set_frequency("DC3", 80_000)
# test_sim = Simulation(
#     "Simp58_101", evaribs
# )
# test_sim.evaluate_center_poly(0, 0, 0)
# test_sim.find_equilib_positions()
# res = test_sim.ion_equilibrium_positions

# for num_ions in range(1, constants.max_ion_in_chain + 1):
#     # Get equilibrium positions

#     # Get Hessian, 3rd, and 4th tensors
#     hessian = test_sim.get_eq_U_hessian(num_ions)
#     third_tensor = test_sim.get_eq_3rd_der_tensor(num_ions)
#     fourth_tensor = test_sim.get_eq_4th_der_tensor(num_ions)

#     # Print the results
#     print(f"Number of ions: {num_ions}")

# for num_ions in range(1, constants.max_ion_in_chain + 1):
#     test_sim.get_mode_eigenvec_and_val(num_ions)

# eigenval, eigenvec = test_sim.get_mode_eigenvec_and_val(1)
# print(test_sim.get_principal_freq_at_min())
# for val in eigenval:
#     print(val)
#     print(math.sqrt(val / constants.ion_mass) / (2 * math.pi))
# # print(test_sim.ion_eigenvectors)
# for i in range(1,1+1):
#     test_sim.get_3_wise_mode_couplings(i)
#     test_sim.get_4_wise_mode_couplings(i)


# print(test_sim.get_2_wise_mode_couplings(1))
# t2 = time.time()

# # for key in res:
# #     print(key)
# #     print(res[key])


# # print(
# #     test_sim.get_U_using_polyfit_dimensionless(
# #         np.array([[-1e-6,0,0],[1e-6,0,0]]).flatten()
# #     )
# # )
# # print(
# #     test_sim.get_U_using_polyfit_dimensionless(
# #         np.array([[-2e-6, 0, 0], [2e-6, 0, 0]]).flatten()
# #     )
# # )
# # print(
# #     test_sim.get_U_using_polyfit_dimensionless(
# #         np.array([[-4e-6, 0, 0], [4e-6, 0, 0]]).flatten()
# #     )
# # )
# print("Time taken: ", t2 - t1)
