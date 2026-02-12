"""
This file is to define the Electrode_vars class and functions to create instances of it.
See Constants.py for the electrode naming convention
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Iterable
import math
import constants  # assumes constants.electrode_names exists
from dataclasses import dataclass, field


def _sanitize_label(label: str) -> str:
    return label.replace(" ", "_").replace("/", "_")


def drive_colname(dk: DriveKey) -> str:
    base = "Static" if dk.f_uHz == 0 else _sanitize_label(dk.label)
    return f"{base}_TotalV"


MICRO = 1_000_000

# ---------- Keys --------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DriveKey:
    """Stable key for a single tone: frequency & phase, quantized for hashing."""

    f_uHz: int
    phi_uRad: int
    # Don't let 'label' affect equality/hash unless you really want it to.
    label: str = field(compare=False)

    @staticmethod
    def from_float(f_hz: float, phi_rad: float, label: str) -> "DriveKey":
        if not isinstance(label, str) or not label.strip():
            raise ValueError("DriveKey label must be a non-empty string.")
        phi = ((phi_rad + math.pi) % (2 * math.pi)) - math.pi
        return DriveKey(round(f_hz * MICRO), round(phi * MICRO), label.strip())

    @property
    def f_hz(self) -> float:
        return self.f_uHz / MICRO

    @property
    def omega(self) -> float:
        return 2 * math.pi * self.f_hz

    @property
    def phi(self) -> float:
        return self.phi_uRad / MICRO


# ---------- Per-drive amplitudes ---------------------------------------------


class Electrode_Amplitudes:
    """
    Holds per-electrode amplitudes in microvolts (ints) for numerical stability.
    Public API uses volts.
    """

    def __init__(self):
        self._elecs = list(constants.electrode_names)
        self.amplitudes = {el: 0 for el in self._elecs}  # base µV ints
        self._pickoff_uV = {el: 0 for el in self._elecs}  # pickoff µV ints

    def set_amplitude_volt(self, electrode: str, A_volt: float) -> None:
        if electrode not in self.amplitudes:
            raise KeyError(f"Unknown electrode '{electrode}'")
        self.amplitudes[electrode] = round(A_volt * MICRO)

    def add_amplitude_volt(self, electrode: str, dA_volt: float) -> None:
        if electrode not in self.amplitudes:
            raise KeyError(f"Unknown electrode '{electrode}'")
        self.amplitudes[electrode] += round(dA_volt * MICRO)

    def add_amplitudes_volt(self, electrodes: list[str], dA_volt: float) -> None:
        for el in electrodes:
            self.add_amplitude_volt(el, dA_volt)

    def compute_pickoff(self, rf_amp_volt: float) -> None:
        """Recompute pickoff (µV) for each electrode from the given RF reference amplitude (V)."""
        mult = constants.electrode_RF_pickoff_amp_multipliers
        for el in self._elecs:
            m = mult.get(el, 0.0)
            self._pickoff_uV[el] = round(m * rf_amp_volt * MICRO)

    def get_amplitude(self, electrode: str) -> float:
        return (self.amplitudes[electrode] + self._pickoff_uV[electrode]) / MICRO

    def get_amplitudes(self) -> dict[str, float]:
        return {
            el: (self.amplitudes[el] + self._pickoff_uV[el]) / MICRO
            for el in self._elecs
        }


# ---------- Master container --------------------------------------------------


class Trapping_Vars:
    """
    Dict-like container: DriveKey -> Electrode_Amplitudes.
    Includes a default DC entry at (f=0, phi=0, label='DC').
    """

    def __init__(self):
        self.electrodes = list(constants.electrode_names)
        self.Var_dict: dict[DriveKey, Electrode_Amplitudes] = {}
        self.by_label: dict[str, DriveKey] = {}
        self.dc_key = DriveKey.from_float(0.0, 0.0, label="DC")
        self.Var_dict[self.dc_key] = Electrode_Amplitudes()
        self.by_label["DC"] = self.dc_key

    # --- Internal helpers -----------------------------------------------------

    def _rf_ref_name(self, ea: Electrode_Amplitudes) -> Optional[str]:
        """Pick an RF reference electrode name for pickoff. Prefer 'RF1', else first starting with 'RF'."""
        if "RF1" in ea.amplitudes:
            return "RF1"
        for el in ea.amplitudes:
            if el.upper().startswith("RF"):
                return el
        return None

    def _update_pickoff_for_drive(self, dk: DriveKey) -> None:
        """Recompute pickoff for a single drive based on its RF reference amplitude."""
        return
        ea = self.Var_dict[dk]
        ref = self._rf_ref_name(ea)
        rf_amp = ea.get_amplitude(ref) if ref is not None else 0.0
        ea.compute_pickoff(rf_amp)

    def _update_pickoff_all(self) -> None:
        return
        """Recompute pickoff for all drives (run after any changes)."""
        for dk in self.Var_dict.keys():
            self._update_pickoff_for_drive(dk)

    # --- Public API: add drives / set amps (now with pickoff refresh) --------

    def add_driving(
        
        self,
        label: str,
        freq: float,
        phase: float,
        amplitudes: Electrode_Amplitudes | dict[str, float],
    ) -> DriveKey:
        """
        freq is taken in as Hz, phase in radians. Ex Freq = 25500000
        """

        label = label.strip()
        if not label:
            raise ValueError("Drive label must be a non-empty string.")
        if label in self.by_label:
            raise ValueError(f"Duplicate drive label '{label}'.")

        dk = DriveKey.from_float(freq, phase, label)

        # guard same (f,phi) already present under a different label
        if dk in self.Var_dict:
            # find the existing key equal in (f,phi)
            for existing_dk in self.Var_dict.keys():
                if existing_dk == dk:
                    if existing_dk.label != label:
                        raise ValueError(
                            f"A drive at f={dk.f_hz} Hz, phi={dk.phi} rad already exists as '{existing_dk.label}'."
                        )
                    break  # same label, falls through (shouldn’t happen since label is new)

        # normalize amplitudes in
        if isinstance(amplitudes, dict):
            ea = Electrode_Amplitudes()
            for el, A in amplitudes.items():
                ea.set_amplitude_volt(el, A)
        else:
            ea = amplitudes

        self.Var_dict[dk] = ea
        self.by_label[label] = dk  # NEW
        self._update_pickoff_for_drive(dk)
        return dk

    def get_drives(self):
        return self.Var_dict.keys()

    def get_drive_amplitudes(self, drive: DriveKey) -> dict[str, float]:
        return self.Var_dict[drive].get_amplitudes()

    # convenience:
    def set_amp(self, drive: DriveKey, electrode: str, A_volt: float) -> None:
        if drive not in self.Var_dict:
            raise ValueError(f"Drive {drive} does not exist.")
        if electrode not in self.Var_dict[drive].amplitudes:
            raise ValueError(f"Electrode {electrode} does not exist.")
        self.Var_dict[drive].set_amplitude_volt(electrode, A_volt)
        self._update_pickoff_for_drive(drive)  # <-- keep pickoff in sync

    def get_amp(self, drive: DriveKey, electrode: str) -> float:
        if drive not in self.Var_dict:
            raise ValueError(f"Drive {drive} does not exist.")
        if electrode not in self.Var_dict[drive].amplitudes:
            raise ValueError(f"Electrode {electrode} does not exist.")
        return self.Var_dict[drive].get_amplitude(electrode)

    def get_drive_by_label(self, label: str) -> DriveKey:
        return self.by_label[label]

    def all_labels(self) -> list[str]:
        return list(self.by_label.keys())

    # --- New: DC-only twist and endcaps --------------------------------------

    def clear_dc_modifications(self) -> None:
        """Clear any DC modifications (twist, endcaps) by resetting all DC amplitudes to zero."""
        ea = self.Var_dict[self.dc_key]
        for el in ea.amplitudes.keys():
            if el.upper().startswith("DC"):
                ea.set_amplitude_volt(el, 0.0)
        self._update_pickoff_all()

    def add_twist_dc(self, twist: float) -> None:
        """
        Apply 'twist' (in volts) to DC electrodes only: subtract from every DC* electrode on the DC drive.
        """
        ea = self.Var_dict[self.dc_key]
        for el in ea.amplitudes.keys():
            if el.upper().startswith("DC"):
                ea.add_amplitude_volt(el, twist)
        for el in ["RF1", "RF2"]:
            if el in ea.amplitudes:
                ea.add_amplitude_volt(el, -twist)
        self._update_pickoff_all()  # keep pickoff fresh for all freqs

    def add_endcaps_dc(
        self, endcaps: float, endcap_electrodes: Optional[list[str]] = None
    ) -> None:
        """
        Add an 'endcaps' (in volts) bias to DC endcap electrodes on the DC drive only.
        If endcap_electrodes is None, default to the classic set if present: DC1, DC5, DC6, DC10.
        """
        ea = self.Var_dict[self.dc_key]
        if endcap_electrodes is None:
            # candidates = {"DC1", "DC5", "DC6", "DC10", "RF11", "RF15", "RF16", "RF20"}
            candidates = {"DC1", "DC5", "DC6", "DC10"}
            endcap_electrodes = [el for el in ea.amplitudes.keys() if el in candidates]
            if not endcap_electrodes:
                # fallback heuristic: treat any 'ENDCAP' named electrodes as endcaps
                endcap_electrodes = [
                    el for el in ea.amplitudes.keys() if "ENDCAP" in el.upper()
                ]
        for el in endcap_electrodes:
            ea.add_amplitude_volt(el, endcaps)
        self._update_pickoff_all()

    def add_endcaps_center_3E(self, endcaps: float, center: float) -> None:
        """
        Add endcaps to DC1,3,4,6,7,9,10,12 and center to DC2,5,8,11 on the DC drive.
        """
        ea = self.Var_dict[self.dc_key]
        endcap_electrodes = ["DC1", "DC3", "DC4", "DC6", "DC7", "DC9", "DC10", "DC12"]
        center_electrodes = ["DC2", "DC5", "DC8", "DC11"]
        for el in endcap_electrodes:
            if el in ea.amplitudes:
                ea.add_amplitude_volt(el, endcaps)
        for el in center_electrodes:
            if el in ea.amplitudes:
                ea.add_amplitude_volt(el, center)
        self._update_pickoff_all()

    def add_endcaps_mid_center_5E(
        self, endcaps: float, mid: float, center: float
    ) -> None:
        """
        Add endcaps to DC1,5,6,10,11,15,16,20; mid to DC3,8,13,18;
        center to DC2,4,7,9,12,14,17,19 on the DC drive.
        """
        ea = self.Var_dict[self.dc_key]
        endcap_electrodes = ["DC1", "DC5", "DC6", "DC10", "DC11", "DC15", "DC16", "DC20"]
        center_electrodes = ["DC3", "DC8", "DC13", "DC18"]
        mid_electrodes = ["DC2", "DC4", "DC7", "DC9", "DC12", "DC14", "DC17", "DC19"]
        for el in endcap_electrodes:
            if el in ea.amplitudes:
                ea.add_amplitude_volt(el, endcaps)
        for el in mid_electrodes:
            if el in ea.amplitudes:
                ea.add_amplitude_volt(el, mid)
        for el in center_electrodes:
            if el in ea.amplitudes:
                ea.add_amplitude_volt(el, center)
        self._update_pickoff_all()

    def apply_dc_twist_endcaps(
        self,
        twist: float,
        endcaps: float,
        endcap_electrodes: Optional[list[str]] = None,
    ) -> None:
        """Convenience: apply both DC twist and endcaps in one call."""
        self.add_twist_dc(twist)
        self.add_endcaps_dc(endcaps, endcap_electrodes=endcap_electrodes)


if __name__ == "__main__":
    tv = Trapping_Vars()
    # Add one RF drive so pickoff has a reference
    rf = tv.add_driving("RF", 20e6, 0.0, {"RF1": 300.0})
    before = tv.get_drive_amplitudes(tv.dc_key)

    tv.apply_dc_twist_endcaps(twist=-2, endcaps=3)  # volts
    after = tv.get_drive_amplitudes(tv.dc_key)
    print("[demo] DC before -> after (showing first few):")
    for k in list(after.keys())[:12]:
        print(f"  {k}: {before[k]:.6f} -> {after[k]:.6f}")
