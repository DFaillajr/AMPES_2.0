# MIT License
#
# Copyright (c) 2025 David P. Failla Jr., Chuyen J. Nguyen
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#######################################################################################################################
# Code Description
# The Additive Manufacturing Process Event Series generator, AMPES, is a Python-based code for developing an event
# series to be used with numerical simulation work. AMPES leverages open-access Python modules and g-code slicing
# software to create an event series to represent the heat source following a tool path or laser path of a given
# additive manufacturing (AM) process. This allows for the capturing of raster scanning effects inherent of most AM
# processes within a thermomechanical modeling framework. While initially developed for use with laser-powder bed
# fusion (L-PBF), AMPES has been extended for usage with blown-powder laser directed energy deposition (L-DED), wire
# arc directed energy deposition (WA-DED), and fused deposition modeling (FDM).
#######################################################################################################################
# Written by:
# David P. Failla Jr.
# Chuyen J. Nguyen
#######################################################################################################################
# Import necessary packages
import csv
import datetime
import os
import argparse
import yaml
import numpy as np
import shutil
import math
import re
from itertools import chain
from sys import exit
from tqdm import tqdm


# Functions
def perturb(input_arr, dev, type='gaussian'):
    """
    # This is a function to perturb an array of values by a given amount. Currently supports a gaussian/normal
    # distribution with type='gaussian', a strict +/- with type='strict', and uniform with type='uniform'.
    :param input_arr: array or list of power values
    :param dev: deviation to add or subtract from the power
    :param type: type of perturbation. options are gaussian/normal distribution around the dev value or strict
    for +/- dev value
    :return: perturbed array
    """
    arr = np.array(input_arr)
    vals = 0

    rng = np.random.default_rng()
    if type == 'gaussian':
        vals = rng.normal(scale=1.0, size=arr.shape)
        vals *= dev
        vals[input_arr == 0] = 0
    elif type == 'strict':
        vals = rng.integers(low=-1, high=2, size=arr.shape)
        vals *= dev
        vals[input_arr == 0] = 0
    elif type == 'uniform':
        vals = rng.uniform(low=-1, high=1.0, size=arr.shape)
        vals *= dev
        vals[input_arr == 0] = 0
    else:
        vals = np.zeros(arr.shape)

    return arr + vals


def verify_config_var_types(input: dict, types: dict, tracker: list = None):
    """
    Recursively iterates through items of the input config dictionary and verifies them against the given types.
    """
    get_key_recursion = lambda a: " in ".join("'{}'".format(x)
                                              for x in reversed(a))

    def get_type_strings(types):
        type_str = "("
        if type(types) == tuple:
            type_str += ", ".join(x.__name__ for x in types)
        else:
            type_str += types.__name__
        type_str += ")"
        return type_str

    if tracker is None:
        tracker = list()
    for key in input.keys():
        if isinstance(input[key], dict):
            tracker.append(key)
            verify_config_var_types(input[key], types, tracker)
            del tracker[-1]
        else:
            tracker.append(key)
            if key in types.keys():
                if not isinstance(input[key], types[key]):
                    if types[key] is float and isinstance(input[key], int):
                        print("Warning: {} is int, expected float".format(
                            get_key_recursion(tracker)))
                    else:
                        raise TypeError(
                            "TypeError: Config variable {} is not expected type {}"
                            .format(get_key_recursion(tracker),
                                    get_type_strings(types[key])))
            del tracker[-1]


def handle_cond_var(cond_var_key, val_var_key, conf_dict):
    """
    Handles a conditional variable within the configuration file
    """
    try:
        cond_var = conf_dict[cond_var_key]
        if cond_var:
            try:
                val_var = conf_dict[val_var_key]
            except KeyError as e:
                raise KeyError("'{}' is required if '{}' is true".format(
                    val_var_key, cond_var_key))
        else:
            val_var = None
        return cond_var, val_var
    except KeyError as e:
        raise e


def get_idx_from_ranges(num: int, ranges: list):
    for i in range(len(ranges)):
        if num in ranges[i]:
            return i
    return False


# Constants
READING_SECTION_STRING = "Reading g-code file"
ES_SECTION_STRING = "Populating event series output"
DWELL_SECTION_STRING = "Adjusting output times for dwell"

config_var_types = {
    "layer_groups": dict,
    "interval": int,
    "base_speed": (int, float),
    "output_speed": (int, float),
    "layers": list,
    "power": (int, float),
    "interlayer_dwell": (int, float),
    "layer_height": (int, float),
    "last_layer_height_change": (int, float),
    "substrate": float,
    "xorg_shift": (int, float),
    "yorg_shift": (int, float),
    "zorg_shift": (int, float),
    "dwell": bool,
    "roller": bool,
    "w_dwell": (int, float),
    "roller_height_offset": (int, float),
    "power_fluctuation": bool,
    "deviation": (int, float),
    "scheme": str,
    "comment_event_series": bool,
    "comment_string": str,
    "process_param_request": bool,
    "time_series": bool,
    "time_series_sample_points": int
}

power_fluc_schemes = ["gaussian", "strict", "uniform"]

pattern = re.compile(r"[XYZFE]-?\d+\.?\d*(e[\-\+]\d*)?"
                    )  # matching pattern for coordinate strings

# Argument Parsing
cwd = os.getcwd()
parser = argparse.ArgumentParser(
    prog="AM Tech",
    description=
    "Reads a RepRap gcode file and outputs Abaqus compatible .inp files based off of it."
)
parser.add_argument("-i",
                    "--input_gcode",
                    help="Location of the gcode file to use as input")
parser.add_argument(
    "-c",
    "--config",
    help=
    "Config input YAML that will be used to initialize parameters, defaults to <current working dir>/input.yaml",
    default=os.path.join(cwd, "input.yaml"))
parser.add_argument(
    "-d",
    "--output_dir",
    help=
    "Directory to output files to, defaults to <current working dir>/output",
    default=os.path.join(cwd, "output"))
parser.add_argument(
    "-o",
    "--outfile_name",
    help="Basename for the files that will be outputted, defaults to \"output\"",
    default="output")
parser.add_argument(
    "--headless",
    help="Option to run AMPES in headless mode",
    action='store_true'
)

args = parser.parse_args()

headless = args.headless

# Process Parameters
# All are set from the config provided as an argument

# Load in config yaml file.
if os.path.isfile(args.config):
    try:
        with open(args.config, "r") as config_file:
            config = yaml.safe_load(config_file)
    except Exception as e:
        print(e)
        print('')
        exit("Error: Could not load input YAML file, given {}".format(
            args.config))
else:
    exit("Error: Config YAML file passed was not found, given {}".format(
        args.config))

try:
    verify_config_var_types(config, config_var_types)
except TypeError as e:
    print(e)
    exit("Error: Expected variable type errors found in configuration file")

try:
    layer_groups = config["layer_groups"]
    interval = config["interval"]
    layer_height = config["layer_height"]
    last_layer_height_change = config["last_layer_height_change"]
    dwell = config["dwell"]
    roller, w_dwell = handle_cond_var("roller", "w_dwell", config)
    roller_height_offset = config["roller_height_offset"]
    process_param_request = config["process_param_request"]
    substrate = config["substrate"]
    time_series, time_series_sample_points = handle_cond_var(
        "time_series", "time_series_sample_points", config)
    #sample_point_count = config["sample_point_count"] not yet implemented
    power_fluc, deviation = handle_cond_var("power_fluctuation", "deviation",
                                            config)
    _, scheme = handle_cond_var("power_fluctuation", "scheme", config)
    comment_event_series, comment_string = handle_cond_var(
        "comment_event_series", "comment_string", config)
    # org_shift used if event series origin is not the same as mesh origin
    xorg_shift = config["xorg_shift"]
    yorg_shift = config["yorg_shift"]
    zorg_shift = config["zorg_shift"]

    # handle optional precision variables
    if "es_precision" in config.keys():
        es_precision = config["es_precision"]
    else:
        es_precision = 6
    if "ts_precision" in config.keys():
        ts_precision = config["ts_precision"]
    else:
        ts_precision = 2

    # handle power fluctuation scheme validity
    if power_fluc:
        if scheme not in power_fluc_schemes:
            exit("Error: Scheme given '{}' not in list of valid schemes {}".
                 format(scheme, power_fluc_schemes))

    # handle roller being true but roller time set to 0
    if roller:
        if w_dwell <= 0:
            exit("Error: Roller must be a positive nonzero number but is {}".
            format(w_dwell))

    # warn for time points demanded, but being 0
    if time_series:
        if time_series_sample_points == 0:
            print(
                "Warning: Time series requested, but number of time points requested between layers is 0"
            )

except KeyError as e:
    exit("Error: Config YAML does not contain expected key: {}".format(
        e.args[0]))

# process layer_groups variable
try:
    layer_group_keys = layer_groups.keys()
    layer_group_list = list(layer_groups.values())
    group_flag = False  # boolean flag for if there is more than one group
    if len(layer_group_list) > 1 or "layers" in layer_group_list[0].keys():
        # get ranges from the intervals in layer_groups
        group_flag = True
        intervals = [
            range(layer_group["layers"][0] - 1, layer_group["layers"][1] + 1)
            for layer_group in layer_group_list
        ]

        # use first layer group's speeds to determine f-value speeds to check
        first_layer_group = layer_group_list[0]
        if "base_speed" not in first_layer_group["infill"].keys() or \
           "base_speed" not in first_layer_group["contour"].keys():
            raise KeyError(
                "The first layer group's sections must contain a 'base_speed' variable set to the corresponding section's speed used within the gcode."
            )
        infill_f_val = first_layer_group["infill"]["base_speed"] * 60
        contour_f_val = first_layer_group["contour"]["base_speed"] * 60

        for layer_group in layer_group_list[1:]:
            if "output_speed" not in layer_group["infill"].keys() or \
               "output_speed" not in layer_group["contour"].keys():
                raise KeyError(
                    "Layer groups' infill and contour sections after first layer group must contain an 'output_speed' variable"
                )

        # handle interlayer dwell time being less than roller time
        if roller:
            for ind in range(len(layer_group_list)):
                if layer_group_list[ind]["interlayer_dwell"] < w_dwell:
                    exit("Error: Layer group '{}' has a interlayer dwell time less than w_dwell. Interlayer dwell is inclusive of roller time and thus should be equal to or greater than the value set for w_dwell.".
                    format(list(layer_group_keys)[ind]))
    else:
        # handles the case of a single layer group
        layer_group = list(layer_groups.values())[0]
        if "base_speed" not in layer_group["infill"].keys() or \
           "base_speed" not in layer_group["contour"].keys():
            raise KeyError(
                "The layer group's sections must contain a 'base_speed' variable set to the corresponding section's speed used within the gcode."
            )
        infill_f_val = layer_group["infill"]["base_speed"] * 60
        contour_f_val = layer_group["contour"]["base_speed"] * 60
        infill_power = layer_group["infill"]["power"]
        contour_power = layer_group["contour"]["power"]
        interlayer_dwell = layer_group["interlayer_dwell"]

        # handle optional output speed values
        if "output_speed" not in layer_group["infill"].keys():
            infill_speed = layer_group["infill"]["base_speed"]
        else:
            infill_speed = layer_group["infill"]["output_speed"]

        if "output_speed" not in layer_group["contour"].keys():
            contour_speed = layer_group["contour"]["base_speed"]
        else:
            contour_speed = layer_group["contour"]["output_speed"]

        # handle interlayer dwell time being less than roller time
        if roller and layer_group["interlayer_dwell"] < w_dwell:
            exit("Error: Interlayer dwell time is less than w_dwell. Interlayer dwell is inclusive of roller time and thus should be equal to or greater than the value set for w_dwell.")

except KeyError as e:
    exit("Error: Layer groups missing expected variable: {}".format(e.args[0]))

# used for recording time to completion
e = datetime.datetime.now()

# initializing arrays and variables
power_out = []
x_out = []
y_out = []
z_out = []
t_out = []
time = 0
z_roller = []
t_roller = []
t_wiper = []
z_wiper = []

# Handle passed arguments

# find file or use user-passed filename
gcode_filename = None
if not args.input_gcode:
    # search current dir for a gcode file
    files = os.listdir(os.getcwd())
    for file in files:
        if os.path.isfile(file) and file[-5:] == "gcode":
            gcode_filename = os.path.join(os.getcwd(), file)
            break
    if not gcode_filename:
        exit(
            "Error: No g-code file passed as argument and could not find g-code file in working directory"
        )
    else:
        print(
            "No g-code file passed as argument. Using {} as g-code file".format(
                gcode_filename))
else:
    gcode_filename = args.input_gcode
    if args.input_gcode[-5:] == "gcode":
        if not os.path.isfile(gcode_filename):
            # Check to see if a gcode file is in the given path
            exit("Error: g-code file passed as input was not found, given {}".
                 format(gcode_filename))
    else:
        exit(
            "Error: File passed as input is not a g-code file (must end with .gcode), given {}"
            .format(gcode_filename))

basename = args.outfile_name
output_dir = args.output_dir
filename_start = os.path.join(output_dir, basename)
main_event_series = filename_start + ".inp"
roller_event_series = filename_start + "_roller.inp"
process_parameter_out = filename_start + "_process_parameter.csv"
time_series_out = filename_start + "_time_series.inp"

if not os.path.isdir(output_dir):
    os.mkdir(output_dir)

# Gcode Reading

with open(gcode_filename, "r") as gcode_file:
    #print("Reading g-code file")
    # more variables needed for reading gcodes
    x = []
    y = []
    z = []
    f = []
    z_posl = []
    power = []
    z_pos = 0
    # removing white spaces on lines with G1 or G0
    if headless:
        print(READING_SECTION_STRING)

    for v in tqdm(range(100), desc="Reading g-code file",ascii=False, ncols=125, disable=headless):
        for line in gcode_file:
            #checks for gcode software to account for initial Z movement
            if line.__contains__("Slic3r") :
                z_start = 0
            elif line.__contains__("Cura") :
                z_start = 1
                z.append(z_start) #modifies z_array if Cura was used to account for removal of initial Z value
                z_posl.append(z_start)
            if line.startswith("G1") or line.startswith(
                    "G0"):  # only reads movement commands
                # Adding 0 infront of decimal point for negative float values less than -1.0
                line = line.replace('-.', '-0.')
                # Replacing ; with single space and splitting into list
                line = line.replace(';', ' ').split()
                # Add coordinates to corresponding arrays # changed linestring here
                for item in line:
                    if pattern.fullmatch(item):
                        if item[0] == "X":
                            x.append(float(item[1:]))
                            f.append(curr_f)
                            z_pos += 1
                        elif item[0] == "Y":
                            y.append(float(item[1:]))
                        elif item[0] == "Z":
                            z.append(float(item[1:]))
                            if group_flag:
                                group_idx = get_idx_from_ranges(
                                    len(z) - 1, intervals)
                                if group_idx == -1:
                                    # break at z value since we don't want to record past a layer jump outside of ranges
                                    break
                            z_posl.append(z_pos)  # Count of z positions per layer
                        elif item[0] == "F":
                            curr_f = float(item[1:])
                if group_flag and group_idx == -1:
                    break
                if "X" in "".join(line):
                    if "E" in "".join(line):
                        if group_flag:
                            infill_power = layer_group_list[group_idx]["infill"][
                                "power"]
                            contour_power = layer_group_list[group_idx]["contour"][
                                "power"]
                        if curr_f == infill_f_val:
                            power.append(infill_power)
                        elif curr_f == contour_f_val:
                            power.append(contour_power)
                        else:
                            exit(
                                "ERROR: g-code contains unexpected F values. Verify that the speed used in g-code file is {} for infill region and {} for contour region."
                                .format(infill_f_val / 60, contour_f_val / 60))
                    else:
                        power.append(0)

#Declaration of Slicing software package used as noted in .gcode file. Slic3r is default
if z_start == 1 :
    print("Cura slicing software used")
else :
   print("Slic3r slicing software used") 

# Using the given velocity and time step, the velocity in each direction is calculated. This is used to determine
# the incremental movement in each direction and then added to the previous value to give the next coordinate. When
# the position command value is reached, it goes to the next value
z_coord = z[1] #changed to 0 9/16 DPF
j = 2
x_out = np.array([x[0]])
y_out = np.array([y[0]])
z_out = np.array([z[1]])
power_out = np.array([0])
t_out = np.array([time])
section_recorder = {
    "indexes": [],
    "type": []
}  # for commenting infill and contour sections
curr_sec = ""  # tracks current section
if headless:
    print(ES_SECTION_STRING)

for i in tqdm(range(1, len(x)), desc="Populating event series output", ascii=False, ncols=125, disable=headless):
    if group_flag:
        # set speed according to input file groups if needed
        group_idx = get_idx_from_ranges(j - 2, intervals)
        if group_idx == -1:
            break

        layer_group = layer_group_list[group_idx]
        if group_idx == 0:
            if "output_speed" not in layer_group["infill"].keys():
                infill_speed = layer_group["infill"]["base_speed"]
            else:
                infill_speed = layer_group["infill"]["output_speed"]

            if "output_speed" not in layer_group["contour"].keys():
                contour_speed = layer_group["contour"]["base_speed"]
            else:
                contour_speed = layer_group["contour"]["output_speed"]
        else:
            infill_speed = layer_group_list[group_idx]["infill"]["output_speed"]
            contour_speed = layer_group_list[group_idx]["contour"][
                "output_speed"]

    del_x = x[i] - x[i - 1]  # incremental change in x
    del_y = y[i] - y[i - 1]  # incremental change in y

    # determining time to move between points based on xy data and velocity
    del_d = math.sqrt(pow(del_x, 2) + pow(del_y, 2))

    if f[i] == infill_f_val:
        vel = infill_speed
        if comment_event_series and curr_sec != "infill":
            section_recorder["indexes"].append(len(x_out) - 1)
            section_recorder["type"].append("infill")
            curr_sec = "infill"
    elif f[i] == contour_f_val:
        vel = contour_speed
        if comment_event_series and curr_sec != "contour":
            section_recorder["indexes"].append(len(x_out) - 1)
            section_recorder["type"].append("contour")
            curr_sec = "contour"

    del_t = del_d / vel
    
    # add interpolated values to output arrays
    tmp_x = np.linspace(x[i - 1], x[i], interval + 2)
    tmp_y = np.linspace(y[i - 1], y[i], interval + 2)
    tmp_z = [z_coord] * (interval + 2)
    tmp_p = [power[i]] * (interval + 2)
    tmp_t = np.linspace(time, time + del_t, interval + 2)
    x_out = np.concatenate([x_out[:-1], tmp_x])
    y_out = np.concatenate([y_out[:-1], tmp_y])
    z_out = np.concatenate([z_out[:-1], tmp_z])
    power_out = np.concatenate([power_out[:-1], tmp_p])
    t_out = np.concatenate([t_out[:-1], tmp_t])
    
    time += del_t

    # Recording gcode converted values of x, y, z, power, and time to output arrays
    # This step occurs to assist the user for reading the output event series. The z-jump
    # could take place with no points in between if desired
    if j < len(z_posl):  #if j is less than the total number of layers in the build
        if i == z_posl[j] - 1:  #if i is equal to one less than the number of z positions of the jth layer
            # peform a z jump of layer_height distance
            del_z = layer_height

            # get current z, add height to it
            curr_z = z_out[-1]
            tmp_z = np.array([curr_z, curr_z + del_z])
            z_out = np.concatenate([z_out, tmp_z])
            # copy x-y values for the jump since it is stationary
            x_out = np.concatenate([x_out, [x_out[-1]] * 2])
            y_out = np.concatenate([y_out, [y_out[-1]] * 2])
            power_out = np.concatenate([power_out, [0] * 2])
            t_out = np.concatenate([t_out, [t_out[-1]] * 2])
            z_coord += layer_height
            j += 1

zmax_temp = max(z_out)
if last_layer_height_change != 0 :
    print("Adjusting last layer height")
    for L in range(len(z_out)) :
        if z_out[L] == zmax_temp :
            z_out[L] = z_out[L] + last_layer_height_change

if dwell or time_series:
    # create layer jump tracking array if required
    z_inc_arr = [(z_posl[i] - 1) * (interval + 1) + (i - 1) * 2 for i in range(2,len(z_posl))]

# Adjust time output array by dwell time variables if option is set
if dwell:
    if headless:
        print(DWELL_SECTION_STRING)
    if roller:
        t_out += w_dwell  # increment whole t_out array by roller time
    for i in tqdm(range(len(z_inc_arr)), desc="Adjusting output times for dwell", ascii=False, ncols=125, disable=headless):
        if group_flag:
            group_idx = get_idx_from_ranges(i, intervals)
            if group_idx == -1:
                break
            interlayer_dwell = layer_group_list[group_idx]["interlayer_dwell"]
        t_out[z_inc_arr[i]:] += interlayer_dwell
else:
    print("Skipping dwell")

# The following develops the wiper event series from position data and constructs the output power array.
# The x and y are fixed to match the AM machine and will have the same wiper characteristics for any print
if roller:
    # initialize roller arrays
    t_roller = [t_out[0]]
    z_roller = list()
    for i in z_inc_arr:
        # when z increases, append the last item
        t_roller.append(t_out[i])
        z_roller.append(z_out[i + 2])

    # end by appending the last item
    t_roller.append(t_out[i])
    z_roller = [z_out[0]] + z_roller

    if dwell:
        # utilize itertools to produce flattened arrays transformed with the expected dwell times
        t_wiper = chain.from_iterable(
            (t_roller[i] - w_dwell,
             t_roller[i]) if i < len(t_roller) - 1 else (None, None)
            for i in range(len(t_roller)))
        z_wiper = chain.from_iterable(
            (z_roller[i], z_roller[i]) for i in range(len(z_roller)))
        # need to truncate extra values from above process
        t_wiper = list(t_wiper)[:-2]
        z_wiper = list(z_wiper)
else:
    print("Skipping roller output")

if power_fluc:
    print("Applying {} scheme to fluctuate power".format(scheme))
    power_out = perturb(power_out, deviation, type=scheme)
else:
    print("Skipping power fluctuation")

# exporting event series
with open(main_event_series, 'w', newline='') as csvfile:
    print("Writing print path event series to {}".format(main_event_series))
    position_writer = csv.writer(csvfile)
    rows = []
    for i in range (len(t_out)) : 
        row = [
            round(t_out[i], es_precision),
            round(x_out[i] + xorg_shift, es_precision),
            round(y_out[i] + yorg_shift, es_precision),
            round(z_out[i] - substrate + zorg_shift, 3),
            round(power_out[i], es_precision)
        ]
        rows.append(row)
    if comment_event_series:
        # for adding comments that separate sections between contour and infill
        section_sep_idxs = np.array(section_recorder["indexes"])
        section_sep_types = np.array(section_recorder["type"])
        # iterate backwards in order to not mess up indexing while inserting values
        for i in range(len(section_sep_idxs) - 1, -1, -1):
            rows.insert(
                section_sep_idxs[i],
                [" ".join([comment_string, section_sep_types[i], "section"])])
    position_writer.writerows(rows)

# exporting wiper/roller event series
if roller:
    print("Writing roller event series to {}".format(roller_event_series))
    with open(roller_event_series, 'w', newline='') as csvfile:
        position_writer = csv.writer(csvfile)
        for i in range(len(z_wiper)):
            if i % 2 == 0:
                row = [
                    round(t_wiper[i], es_precision), -125, 250,
                    round(z_wiper[i] - substrate + zorg_shift + roller_height_offset, 3), 1.0
                ]
                position_writer.writerow(row)
            else:
                row = [
                    round(t_wiper[i], es_precision), 125, 250,
                    round(z_wiper[i] - substrate + zorg_shift + roller_height_offset, 3), 0.0
                ]
                position_writer.writerow(row)

# creating temperature output write times for at the end of the print phase for each layer
if time_series:
    print("Writing time series output to {}".format(time_series_out))
    with open(time_series_out, "w", newline='') as csvfile:
        position_writer = csv.writer(csvfile)
        # write out initial points from heatup time
        if roller:
            position_writer.writerow([round(t_wiper[0],
                                            ts_precision)])  # first deposit
        position_writer.writerow([round(t_out[0],
                                        ts_precision)])  # first power on

        for count, i in enumerate(range(z_inc_arr[0], 0, -1)):
            if power_out[i] != 0:
                break

        first_end_ind = z_inc_arr[0] - count + 1

        if time_series_sample_points > 0:
            # add additional sample points between power start and end for a layer
            if time_series_sample_points == 1:
                sample_inds = [int((first_end_ind) / 2), 0]
            else:
                sample_inds = np.linspace(
                    0, first_end_ind, time_series_sample_points + 2).astype(int)

            for ind in sample_inds[1:-1]:
                position_writer.writerow([round(t_out[ind], ts_precision)])

        position_writer.writerow(
            [round(t_out[z_inc_arr[0] - count + 1],
                   ts_precision)])  # first layer complete

        for i in range(len(z_inc_arr)):
            # t_wiper uses *2 because it has two points per z jump and (i+1) means skip the first two
            #   since those do not correspond to a jump but rather the preheat time
            for start_count, j in enumerate(range(z_inc_arr[i],
                                                  len(power_out))):
                # seek first non-zero value in power array after deposit
                if power_out[j] != 0:
                    break
            start_ind = z_inc_arr[i] + start_count
            power_start = t_out[start_ind]
            # seek layer end using reverse search from next layer jump
            if i < len(z_inc_arr) - 1:
                seek_from_ind = z_inc_arr[i + 1]
            else:
                # required to handle last movement block
                seek_from_ind = len(x_out) - 1

            for end_count, j in enumerate(range(seek_from_ind, 0, -1)):
                # seek first non-zero value in power array prior to next z jump
                if power_out[j] != 0:
                    break
            if end_count != 0:
                # this handles the case that the last point is a power-on point
                end_ind = seek_from_ind - end_count + 1
                power_end = t_out[end_ind]
            else:
                end_ind = seek_from_ind
                power_end = t_out[end_ind]

            if time_series_sample_points > 0:
                # add additional sample points between power start and end for a layer
                if time_series_sample_points == 1:
                    sample_inds = [
                        int((end_ind - start_ind) / 2) + start_ind, 0
                    ]
                else:
                    sample_inds = np.linspace(start_ind, end_ind,
                                              time_series_sample_points +
                                              2).astype(int)

            # write out to file
            if roller:
                deposit_start = t_wiper[(i + 1) * 2]
                position_writer.writerow([round(deposit_start, ts_precision)])
            position_writer.writerow([round(power_start, ts_precision)])
            if time_series_sample_points > 0:
                for ind in sample_inds[1:-1]:
                    position_writer.writerow([round(t_out[ind], ts_precision)])
            position_writer.writerow([round(power_end, ts_precision)])
else:
    print("Skipping coordinate output")

# Exporting process parameters used in this run
if process_param_request:
    print("Writing process parameter csv file to {}".format(
        process_parameter_out))
    with open(process_parameter_out, 'w', newline='') as csvfile:
        position_writer = csv.writer(csvfile)
        date_time = [
            "Developed {} at {}".format(e.strftime("%Y/%d/%m"),
                                        e.strftime("%H:%M"))
        ]
        position_writer.writerow(date_time)
        header = ["Parameter", "Value", "Unit"]
        write_rows = []
        if z_start == 1 :
            write_rows.append(["Cura slicing software used"])
            write_rows.append(["for .gcode file generation"])
        else :
            write_rows.append(["Slic3r slicing software used"]) 
            write_rows.append(["for .gcode file generation"])
        write_rows.append([])
        if not group_flag:
            # determine output velocities
            write_rows.append(["##Print parameters"])
            write_rows.append(header)
            write_rows.append(
                ["Infill Base Velocity", infill_f_val / 60, "mm/s"])
            write_rows.append(["Infill Output Velocity", infill_speed, "mm/s"])
            write_rows.append(["Infill Power", infill_power, "mW"])
            write_rows.append(
                ["Contour Base Velocity", contour_f_val / 60, "mm/s"])
            write_rows.append(
                ["Contour Output Velocity", contour_speed, "mm/s"])
            write_rows.append(["Contour Power", contour_power, "mW"])
            write_rows.append(["Interlayer Dwell Time", interlayer_dwell, "s"])
        else:
            first_group = True
            for group_name, group_params in config["layer_groups"].items():
                write_rows.append([])
                write_rows.append(
                    ["##Layer group print parameters", group_name])
                write_rows.append(header)
                for key, value in group_params.items():
                    if key == "layers":
                        write_rows.append(["Layers in Group", value, "count"])
                    elif key == "infill":
                        infill_output_speed = value[
                            "base_speed"] if "output_speed" not in value.keys(
                            ) else value["output_speed"]
                        if first_group:
                            write_rows.append([
                                "Infill Base Velocity", value["base_speed"],
                                "mm/s"
                            ])
                        write_rows.append([
                            "Infill Output Velocity", infill_output_speed,
                            "mm/s"
                        ])
                        write_rows.append(
                            ["Infill Power", value["power"], "mW"])
                    elif key == "contour":
                        contour_output_speed = value[
                            "base_speed"] if "output_speed" not in value.keys(
                            ) else value["output_speed"]
                        if first_group:
                            write_rows.append([
                                "Contour Base Velocity", value["base_speed"],
                                "mm/s"
                            ])
                        write_rows.append([
                            "Contour Output Velocity", contour_output_speed,
                            "mm/s"
                        ])
                        write_rows.append(
                            ["Contour Power", value["power"], "mW"])
                    elif key == "interlayer_dwell":
                        write_rows.append(["Dwell Time", value, "s"])
                    else:
                        write_rows.append(
                            ["Unexpected item `{}`".format(key), value, "N/A"])
                first_group = False

        if roller:
            write_rows.append([])
            write_rows.append(["##Roller parameters"])
            write_rows.append(header)
            write_rows.append(["Roller time", w_dwell, "s"])

        write_rows.append([])
        write_rows.append(["##Overall parameters"])
        write_rows.append(["Intervals", interval, "#"])
        write_rows.append(["Layer Height", layer_height, "mm"])
        write_rows.append(["Last Layer Height Change", last_layer_height_change, "mm"])
        write_rows.append(["roller_height_offset", roller_height_offset, "mm"])
        write_rows.append(["Substrate Thickness", substrate, "mm"])
        write_rows.append(["Origin Shift in X", xorg_shift, "mm"])
        write_rows.append(["Origin Shift in Y", yorg_shift, "mm"])
        write_rows.append(["Origin Shift in Z", zorg_shift, "mm"])
        for row in write_rows:
            position_writer.writerow(row)
else:
    print("Skipping process parameter output")

print("Complete" + "\n")
