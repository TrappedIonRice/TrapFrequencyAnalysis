"""
This file will contain all nessecary function to turn raw COMSOL data into usable dataframes

For now the plan is as follows:
Comsol --> Raw TXT file per electrode
Raw TXT file per electrode --> Extracted CSV file (with x,y,z,V,Ex,Ey,Ez columns)
12 CSV files per simulation (one for each electrode) --> Combined CSV file (with x,y,z,V,Ex,Ey,Ez columns for all electrodes)
"""

import pandas as pd
import numpy as np
import os
import re
import datetime
import time
import csv
import constants
# import electrode_vars_old as evars


def _read_header_lines(file_path, max_lines=20):
    header_lines = []
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        for _ in range(max_lines):
            line = f.readline()
            if not line:
                break
            header_lines.append(line.strip())
    return header_lines


def _normalize_length_unit(unit):
    unit = unit.strip().lower()
    unit = unit.replace("meters", "m").replace("meter", "m")
    unit = unit.replace("millimeters", "mm").replace("millimeter", "mm")
    unit = unit.replace("centimeters", "cm").replace("centimeter", "cm")
    unit = unit.replace("microns", "um").replace("micron", "um")
    unit = unit.replace("nanometers", "nm").replace("nanometer", "nm")
    return unit


def _length_unit_scale_to_m(unit):
    unit = _normalize_length_unit(unit)
    scale_map = {
        "m": 1.0,
        "mm": 1e-3,
        "cm": 1e-2,
        "um": 1e-6,
        "nm": 1e-9,
    }
    return scale_map.get(unit)


def _infer_length_unit(header_lines):
    # Look for explicit x/y/z unit markers like "x (mm)" or "x [mm]"
    axis_units = {}
    axis_pattern = re.compile(r"\b([xyz])\s*[\[(]\s*([a-zA-Z]+)\s*[\])]")
    for line in header_lines:
        for axis, unit in axis_pattern.findall(line):
            axis_units[axis.lower()] = unit

    if axis_units:
        units = {axis: _normalize_length_unit(u) for axis, u in axis_units.items()}
        unique_units = set(units.values())
        if len(unique_units) == 1:
            return unique_units.pop(), False
        print(f"Warning: mixed axis units found in header: {units}")
        return units.get("x", "mm"), True

    # Look for a general "length unit: mm" style line
    generic_pattern = re.compile(r"(length|coordinate)\s*unit[s]?\s*[:=]\s*([a-zA-Z]+)")
    for line in header_lines:
        match = generic_pattern.search(line.lower())
        if match:
            return _normalize_length_unit(match.group(2)), False

    return "mm", True


def extract_raw_trap_sim_data(file_path):
    """
    Extract raw data from a text file with the following columns: x, y, z, V, Ex, Ey, Ez.
    Note this function is only guaranteed to be correct for data extracted as detailed in the readme

    Parameters:
    file_path (str): The path to the text file.

    Returns:
    pd.DataFrame: DataFrame containing the extracted data.
    also saves this df as a pickle to the same folder as the file_path
    """
    # extract the last bit of the file path to use as the name of the dataframe
    blade_name = os.path.basename(file_path).split(".")[0].split("_")[0]

    # extract the simulation from the file path, meaning the dataset folder name under Data
    path_parts = os.path.normpath(file_path).split(os.sep)
    if "Data" in path_parts:
        data_idx = path_parts.index("Data")
        simulation = path_parts[data_idx + 1] if data_idx + 1 < len(path_parts) else ""
    else:
        simulation = os.path.basename(os.path.dirname(file_path))

    print("Extracting data from " + blade_name + " in " + simulation + " simulation")

    header_lines = _read_header_lines(file_path)
    length_unit, assumed = _infer_length_unit(header_lines)
    print(length_unit)
    length_scale = _length_unit_scale_to_m(length_unit)
    if length_scale is None:
        print(
            f"Warning: Unrecognized length unit '{length_unit}'. Assuming mm."
        )
        length_unit = "mm"
        length_scale = 1e-3
        assumed = True

    if assumed:
        print(
            f"Warning: Could not determine length unit from header. Assuming {length_unit}."
        )

    # Read the file, skipping the metadata lines
    df = pd.read_csv(file_path, sep="\s+", skiprows=9)

    # Assign meaningful column names based on the file's description
    df.columns = [
        "x",
        "y",
        "z",
        "V",
        "Ex",
        "Ey",
        "Ez",
    ]

    # Convert x,y,z to meters using the detected unit
    for column in ["x", "y", "z"]:
        df[column] = df[column] * length_scale

    # #Now convert the Ex,Ey,Ez to standard SI units
    # for column in ["Ex", "Ey", "Ez"]:
    #     df[column] = df[column]

    # #and now for V
    # df["V"] = df["V"]

    # Now we will iterate through all the columns and round the values to a specified number of decimal places
    for column in df.columns:
        df[column] = df[column].round(12)

    # now we will find the dimensions of the data, meaning how many distinct x, y, z values are there and how sperated each axis's sampling is
    # we will return the dimensions as a tuple
    x_dimension = len(df["x"].unique())
    y_dimension = len(df["y"].unique())
    z_dimension = len(df["z"].unique())

    x_spacing = round(float(df["x"].unique()[1] - df["x"].unique()[0]), 8)
    y_spacing = round(float(df["y"].unique()[1] - df["y"].unique()[0]), 8)
    z_spacing = round(float(df["z"].unique()[1] - df["z"].unique()[0]), 8)

    dimension = (
        ("x_dim", x_dimension),
        ("x_spacing", x_spacing),
        ("y_dim", y_dimension),
        ("y_spacing", y_spacing),
        ("z_dim", z_dimension),
        ("z_spacing", z_spacing),
    )

    # add the dimension as a tupple to the dataframe under the name "dim"
    df.attrs["dim"] = dimension
    df.attrs["length_unit"] = length_unit
    df.attrs["length_unit_scale_to_m"] = length_scale

    df.to_pickle(
        "C:\\Users\\TrappedIonRiceDell2\\Documents\\GitHub\\TrapFrequencyAnalysis\\Data\\"
        + simulation
        + "\\"
        + blade_name
        + "_extracted.csv"
    )

    return df


def make_simulation_dataframe(folder_path):
    """
    Create a dataframe from all the extracted data files in a given sim.
    (with x,y,z,V,Ex,Ey,Ez columns for all electrodes)

    Parameters:
    folder_path (str): The path to the folder containing the extracted data files.

    Returns:
    pd.DataFrame: DataFrame containing the combined data from all files.
    also saves the df as a pickle to folder_path
    """

    # for each txt file in folder_path check if the corresponding csv file exists, if it does skip this txt file
    # if it does not exist, extract the data from the txt file and save it as a csv file using the extract_raw_trap_sim_data function
    for file in os.listdir(folder_path):
        if file.endswith(".txt"):
            csv_file = os.path.join(
                folder_path, file.replace("_Raw.txt", "_extracted.csv")
            )
            if not os.path.exists(csv_file):
                extract_raw_trap_sim_data(os.path.join(folder_path, file))

    # Now we shall merge these dataframes into a single dataframe
    # This data frame columns will be ["x", "y", "z", and then for each electrode in the simulation, "V", "Ex", "Ey", "Ez" for each electrode] ex: RF1_V, RF1_Ex, RF1_Ey, RF1_Ez, etc.
    # The data frame will also have a column for TotalV
    # We will also give the dataframe an atrabute called electrode_vars, which will be None for now

    # init the dataframe
    df = pd.DataFrame()
    # get the list of all csv files in the folder
    csv_files = [f for f in os.listdir(folder_path) if f.endswith("_extracted.csv")]
    # print(f"Found {len(csv_files)} csv files in {folder_path}")
    # print(f"csv files: {csv_files}")
    # for each csv file, read it and append it to the dataframe
    names_of_electodes = []  # to keep track of electrode names for later use
    length_units = {}
    for csv_file in csv_files:
        # Extract the electrode name from the file name
        electrode_name = os.path.basename(csv_file).split("_")[
            0
        ]  # Adjust the split logic based on your file naming convention

        # read the csv file
        file_path = os.path.join(folder_path, csv_file)
        temp_df = pd.read_pickle(file_path)

        if temp_df.attrs.get("length_unit") is not None:
            length_units[electrode_name] = temp_df.attrs.get("length_unit")

        # Rename the columns for V, Ex, Ey, Ez
        temp_df.rename(
            columns={
                "V": f"{electrode_name}_V",
                "Ex": f"{electrode_name}_Ex",
                "Ey": f"{electrode_name}_Ey",
                "Ez": f"{electrode_name}_Ez",
            },
            inplace=True,
        )

        names_of_electodes.append(electrode_name)  # add the electrode name to the list

        # Merge it with the main dataframe on x, y, z columns using inner join
        if df.empty:
            df = temp_df
        else:
            df = pd.merge(df, temp_df, on=["x", "y", "z"], how="inner")

    df["TotalV"] = np.nan

    # Weirdness: so far we only work with geometrys of one kind if this changes and this truly becomes a varibale electrode_vars has to change quite a bit

    # Add custom attributes using the attrs property
    df.attrs["electrode_names"] = names_of_electodes
    # df.attrs["electrode_vars"] = evars.Electrode_vars()
    df.attrs["name"] = os.path.basename(folder_path)
    if length_units:
        unique_units = set(length_units.values())
        if len(unique_units) > 1:
            print(f"Warning: Mixed length units detected: {length_units}")
        df.attrs["length_unit"] = next(iter(unique_units))

    # Save the combined dataframe as a pickle file
    df.to_pickle(os.path.join(folder_path, "combined_dataframe.csv"))

    # print the number of points in the dataframe
    print(f"Number of points in the dataframe: {len(df)}")
    print(f"Dataframe shape: {df.shape}")

    return df


# *#*#*# Just for testing #*#*#*#

def get_val_from_point(dataframe, x, y, z, val):
    """
    Get the electric field value (V) at a specific point (x, y, z) from the dataframe.

    Args:
        dataframe (pd.DataFrame): The dataframe containing the data.
        x (float): The x-coordinate of the point.
        y (float): The y-coordinate of the point.
        z (float): The z-coordinate of the point.

    Returns:
        float: The electric field value (V) at the specified point, or a large negative value if the point is not found.
    """
    # Filter the dataframe to find the exact point
    filtered_df = dataframe[
        (dataframe["x"] == x) & (dataframe["y"] == y) & (dataframe["z"] == z)
    ]

    if not filtered_df.empty:
        # Return the V value if the point is found
        return filtered_df.iloc[0][val]
    else:
        # Return a large negative value if the point is not found
        return -1e6


def get_V_from_point(dataframe, x, y, z):
    return get_val_from_point(dataframe, x, y, z, "V")


def get_Ex_from_point(dataframe, x, y, z):
    return get_val_from_point(dataframe, x, y, z, "Ex")


def get_Ey_from_point(dataframe, x, y, z):
    return get_val_from_point(dataframe, x, y, z, "Ey")


def get_Ez_from_point(dataframe, x, y, z):
    return get_val_from_point(dataframe, x, y, z, "Ez")


def get_all_from_point(dataframe, x, y, z):
    # Use the query method for efficient filtering
    print("geting points from data frame")
    filtered_df = dataframe.query("x == @x and y == @y and z == @z")

    if not filtered_df.empty:
        # Directly access the values using iat for better performance
        return [
            filtered_df.iat[0, dataframe.columns.get_loc("V")],
            filtered_df.iat[0, dataframe.columns.get_loc("Ex")],
            filtered_df.iat[0, dataframe.columns.get_loc("Ey")],
            filtered_df.iat[0, dataframe.columns.get_loc("Ez")],
        ]
    else:
        # Return a large negative value if the point is not found
        return -1e6

    # return [
    #     get_V_from_point(dataframe, x, y, z),
    #     get_Ex_from_point(dataframe, x, y, z),
    #     get_Ey_from_point(dataframe, x, y, z),
    #     get_Ez_from_point(dataframe, x, y, z),
    # ]


def get_set_of_points(dataframe):
    return set(zip(dataframe["x"], dataframe["y"], dataframe["z"]))


# *#*#*##*#*#*##*#*#*##*#*#*##*#*#*#

if __name__ == "__main__":
    print("running")
    make_simulation_dataframe(r"Data\Comsol_125")
    #make_simulation_dataframe(r"Data\twodTrap_1")
