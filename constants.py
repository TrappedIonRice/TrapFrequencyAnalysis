"""
This file holds all physical and semantic constants

Electode Naming Diagram:

    Y,Z Plane: 
        DC_Blade_A--------RF2
        |                  |
        |                  |    (+Z dir is up)
        |                  |    (+Y dir is right)
        |                  |
        RF1-----------DC_Blade_B

    DC Blades from -X to +X:
        DC_Blade_A ==> --- DC5  DC4  DC3  DC2  DC1 ---
        DC_Blade_B ==> --- DC6  DC7  DC8  DC9  DC10 ---
        
    RF Blades from -X to +X:
        RF1 ==> --- RF11  RF12  RF13  RF14  RF15 ---
        RF2 ==> --- RF20  RF19  RF18  RF17  RF16 ---

"""

# TODO make this read from a config file ST a UI could eazily edit

import math

import numpy as np

INCLUDE_ALL_RF_PSEUDOPOTENTIALS = False  # False keeps current behavior

ion_mass = 2.885 * (10 ** (-25))  # kg Yb+
# ion_mass = 6.642065 * (10 ** (-26))  # kg Ca40
# ion_mass = 1.5e-26  # kg be
ion_charge = 1.60217662 * (10 ** (-19))  # C
epsilon_0 = 8.854187817e-12  # F/m

# Order: DCs first (unchanged), then legacy combined RFs, then the 10 RF segments
electrode_names = (
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
    "DC11",
    "DC12",
    "DC13",
    "DC14",
    "DC15",
    "DC16",
    "DC17",
    "DC18",
    "DC19",
    "DC20",
    "RF1",
    "RF2",
    "RF11",
    "RF12",
    "RF13",
    "RF14",
    "RF15",
    "RF16",
    "RF17",
    "RF18",
    "RF19",
    "RF20",
)

# Convenience groupings for segmented RF blades
RF1_SEGMENTS = ("RF11", "RF12", "RF13", "RF14", "RF15")  # -X -> +X
RF2_SEGMENTS = (
    "RF16",
    "RF17",
    "RF18",
    "RF19",
    "RF20",
)  # -X -> +X (note: geometry shows 20..16 left->right)
RF_SEGMENTS = RF1_SEGMENTS + RF2_SEGMENTS

DC_ELECTRODES = tuple(f"DC{i}" for i in range(1, 21))
RF_ELECTRODES = ("RF1", "RF2") + RF_SEGMENTS

## used in both directions

# ## 1 D ##
center_region_x_um = 100  # microns
center_region_y_um = 20  # microns
center_region_z_um = center_region_y_um

## 2 D ##
# center_region_x_um = 100  # microns
# center_region_y_um = 25  # microns
# center_region_z_um = 25

# Reference RF frequency used for A-matrix construction (Hz)
RF_FREQ_REF_HZ = 43e6
# Use omega in MHz units for s: omega_MHz = 2*pi*(f_Hz/1e6)
# This is omega/1e6 (rad/s scaled to MHz units), used in s definition.
RF_OMEGA_REF_MHZ = 2.0 * np.pi * (RF_FREQ_REF_HZ / 1e6)
RF_S_MAX_DEFAULT = (600.0**2) / (RF_OMEGA_REF_MHZ**2)

# Nondimensionalization length scale for inverse pipeline
ND_L0_M = 10e-6
ND_L0_UM = 10.0

INVERSE_APP_TRAP_REGISTRY = {
    "2Dtrap_125_45deg_200exp": {
        "display_label": "2Dtrap_125_45deg_200exp",
        "trap_name": "2Dtrap_125_45deg_200exp",
        "dc_electrodes": tuple(f"DC{i}" for i in range(1, 21)),
        "rf_dc_electrodes": ("RF1", "RF2"),
    },
    "simp58_101_GEN3": {
        "display_label": "simp58_101_GEN3",
        "trap_name": "simp58_101",
        "dc_electrodes": tuple(f"DC{i}" for i in range(1, 11)),
        "rf_dc_electrodes": ("RF1", "RF2"),
    },
    "1252dTrapRice_old": {
        "display_label": "1252dTrapRice_old",
        "trap_name": "1252dTrapRice",
        "dc_electrodes": tuple(f"DC{i}" for i in range(1, 21)),
        "rf_dc_electrodes": ("RF1", "RF2"),
    },
    "InnTrapFine": {
        "display_label": "InnTrapFine",
        "trap_name": "InnTrapFine",
        "dc_electrodes": tuple(f"DC{i}" for i in range(1, 13)),
        "rf_dc_electrodes": ("RF1", "RF2"),
    },
}
INVERSE_APP_TRAP_OPTIONS = tuple(INVERSE_APP_TRAP_REGISTRY.keys())



# TODO: The values should be applied differently, the current implemetation is flawed
# What should happen is that each pairwise combination of electodes has a capacitance and then
# that should be used to calculate the pickoff multiplier
# what we have right now is the RF to [blank] pickoffs
# ideally we have a 12 by 12 matrix of capacitances (Should be symmetric)
# trap_capcitence_per_electrode_PF = {
#     "DC1": 0.155,
#     "DC2": 0.068,
#     "DC3": 0.125,
#     "DC4": 0.075,
#     "DC5": 0.157,
#     "DC6": 0.157,
#     "DC7": 0.076,
#     "DC8": 0.125,
#     "DC9": 0.068,
#     "DC10": 0.155,
#     "RF1": 0.01,
#     "RF2": 0.01,
# }

# zeroing for now
trap_capcitence_per_electrode_PF = {
    "DC1": 0.0005,
    "DC2": 0.0005,
    "DC3": 0.0005,
    "DC4": 0.0005,
    "DC5": 0.0005,
    "DC6": 0.0005,
    "DC7": 0.0005,
    "DC8": 0.0005,
    "DC9": 0.0005,
    "DC10": 0.0005,
    "RF1": 0.0005,
    "RF2": 0.0005,
}

trap_capcitence_per_electrode_PF.update(
    {
        "RF11": 0.0002,
        "RF12": 0.0002,
        "RF13": 0.0002,
        "RF14": 0.0002,
        "RF15": 0.0002,
        "RF16": 0.0002,
        "RF17": 0.0002,
        "RF18": 0.0002,
        "RF19": 0.0002,
        "RF20": 0.0002,
    }
)

ground_capacitor_PF = 1000  # PF

electrode_RF_pickoff_amp_multipliers = {}
for electrode in trap_capcitence_per_electrode_PF:
    electrode_RF_pickoff_amp_multipliers[electrode] = trap_capcitence_per_electrode_PF[
        electrode
    ] / (trap_capcitence_per_electrode_PF[electrode] + ground_capacitor_PF)

# electrode_RF_pickoff_amp_multipliers = {
#     **{el: 0.0 for el in DC_ELECTRODES},
#     "RF1": 0.0,
#     "RF2": 0.0,
#     **{el: 0.0 for el in RF_SEGMENTS},
# }


def freq_calcualtion(secondderivative):
    return (
        math.copysign(1, secondderivative)
        * math.sqrt((ion_charge / ion_mass) * abs(secondderivative))
        / (2 * math.pi)
    )


ion_electrode_dis = 0.00025

hbar = 1.0545718e-34  # J*s

max_ion_in_chain = 100

coulomb_constant = 8.9875517873681764 * (10**9)  # N m^2 / C^2


# Should be a flat list/array of length 3n
class _IonInitialGuessDict(dict):
    def __missing__(self, num_ions):
        if num_ions <= 0:
            raise ValueError(f"num_ions must be positive, got {num_ions}")
        if num_ions == 1:
            positions = [[0.0, 0.0, 0.0]]
            self[num_ions] = positions
            return positions

        total_length_m = (num_ions / 2.0) * 1e-6
        x_vals_m = np.linspace(-total_length_m / 2.0, total_length_m / 2.0, num_ions)
        positions = [
            [float(x / length_harmonic_approximation), 0.0, 0.0] for x in x_vals_m
        ]
        self[num_ions] = positions
        return positions


ion_locations_intial_guess = _IonInitialGuessDict()
ion_locations_bounds_1d = {}
ion_locations_bounds_2d = {}
USE_1D_ION_BOUNDS = False

# Length in harmonic apoximation
Z = 1
typical_axial_freq = 225000 * 6.28  # Hz ((Rad/sec))
length_harmonic_approximation = (
    (Z**2 * ion_charge**2) / (4 * 3.1416 * epsilon_0 * ion_mass * typical_axial_freq**2)
) ** (1 / 3)

print("Length in harmonic approximation: ", length_harmonic_approximation)

# Intial equilib positions normallized by l:
ion_locations_intial_guess[1] = [[0, 0, 0]]
ion_locations_intial_guess[2] = [[-0.62996, 0, 0], [0.62996, 0, 0]]
ion_locations_intial_guess[3] = [[-1.3772, 0, 0], [0, 0, 0], [1.3772, 0, 0]]
ion_locations_intial_guess[4] = [
    [-1.4368, 0, 0],
    [-0.55438, 0, 0],
    [-0.55438, 0, 0],
    [1.4368, 0, 0],
]
ion_locations_intial_guess[5] = [
    [-1.7429, 0, 0],
    [-0.8221, 0, 0],
    [0, 0, 0],
    [0.8221, 0, 0],
    [1.7429, 0, 0],
]
ion_locations_intial_guess[6] = [
    [-1.9, 0, 0],
    [-1.1, 0, 0],
    [-0.3, 0, 0],
    [0.3, 0, 0],
    [1.1, 0, 0],
    [1.9, 0, 0],
]
ion_locations_intial_guess[7] = [
    [-2.0, 0, 0],
    [-1.3, 0, 0],
    [-0.7, 0, 0],
    [0, 0, 0],
    [0.7, 0, 0],
    [1.3, 0, 0],
    [2.0, 0, 0],
]
ion_locations_intial_guess[8] = [
    [-2.1, 0, 0],
    [-1.5, 0, 0],
    [-0.9, 0, 0],
    [-0.3, 0, 0],
    [0.3, 0, 0],
    [0.9, 0, 0],
    [1.5, 0, 0],
    [2.1, 0, 0],
]
ion_locations_intial_guess[9] = [
    [-2.2, 0, 0],
    [-1.7, 0, 0],
    [-1.1, 0, 0],
    [-0.5, 0, 0],
    [0, 0, 0],
    [0.5, 0, 0],
    [1.1, 0, 0],
    [1.7, 0, 0],
    [2.2, 0, 0],
]
ion_locations_intial_guess[10] = [
    [-2.3, 0, 0],
    [-1.9, 0, 0],
    [-1.3, 0, 0],
    [-0.7, 0, 0],
    [-0.1, 0, 0],
    [0.1, 0, 0],
    [0.7, 0, 0],
    [1.3, 0, 0],
    [1.9, 0, 0],
    [2.3, 0, 0],
]

## 1 d ##
radial_bounds = 5e-6 / length_harmonic_approximation
axial_bounds = 200e-6 / length_harmonic_approximation
center_x_bounds = (center_region_x_um * 1e-6) / length_harmonic_approximation
center_y_bounds = (center_region_y_um * 1e-6) / length_harmonic_approximation
center_z_bounds = (center_region_z_um * 1e-6) / length_harmonic_approximation

for i in range(1, max_ion_in_chain + 1):
    ion_locations_bounds_1d[i] = [
        (
            -axial_bounds,
            axial_bounds,
        ),
        (-radial_bounds, radial_bounds),
        (-radial_bounds, radial_bounds),
    ] * (i)
    ion_locations_bounds_2d[i] = [
        (-center_x_bounds, center_x_bounds),
        (-center_y_bounds, center_y_bounds),
        (-center_z_bounds, center_z_bounds),
    ] * (i)

    # make sure the initial guess is a float
    for pnt in range(len(ion_locations_intial_guess[i])):
        ion_locations_intial_guess[i][pnt][0] = float(
            ion_locations_intial_guess[i][pnt][0]
        )

ion_locations_bounds = (
    ion_locations_bounds_1d if USE_1D_ION_BOUNDS else ion_locations_bounds_2d
)


# for i in range(1, max_ion_in_chain + 1):
#     for pnt in range(len(ion_locations_intial_guess[i])):
#         ion_locations_intial_guess[i][pnt][0] = (
#             ion_locations_intial_guess[i][pnt][0] * length_harmonic_approximation
#         )
#     ion_locations_bounds[i] = [
#         (-200e-6, 200e-6),
#         (-1e-6, 1e-6),
#         (-1e-6, 1e-6),
#     ] * (i)

# print("Constants loaded")


# testing

# print("Ion locations intial guess: ", ion_locations_intial_guess[6])
# print("Ion locations intial guess: ", ion_locations_intial_guess[7])
# print("Ion locations intial guess: ", ion_locations_intial_guess[8])
# print("Ion locations intial guess: ", ion_locations_intial_guess[9])
# print("Ion locations intial guess: ", ion_locations_intial_guess[10])
# # print("Ion locations bounds: ", ion_locations_bounds)
# init_guess_flatened = np.array(ion_locations_intial_guess[3]).flatten()
# print("init_guess_flatened: ", init_guess_flatened)
