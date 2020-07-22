import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata

import sys


class Measurement:
    def __init__(self, data={}):
        self.data = data

    def get_field(self, field_name=""):
        return self.data[field_name]


class Scan:
    def __init__(self, header, chip_id):
        self.header = header
        self.chip_id = chip_id
        self.measurements = []


def get_scans_from_file(input_file):

    chip_id = ""

    scans = []
    current_scan = None
    field_names = []
    with open(input_file, "r") as infile:
        for iline, line in enumerate(infile):
            line = line.strip()
            if "RD53B" in line:
                if current_scan is not None:
                    scans.append(current_scan)
                chip_id = line.split()[1].split("-")[-1]
                header_data = line.split()[
                    -1
                ]  # the last portion of the header tag has the entiries separated by "_"
                header_data = header_data.split("_")
                header = {}
                for entry in header_data:
                    key, val = entry.split(":")
                    header[key] = val
                current_scan = Scan(header, chip_id)
                continue

            # we assume that the measured quantities stay fixed
            # column separated data is assumed!
            if "I_lim" in line:
                field_names = line.split(",")
                continue

            data_fields = [float(x) for x in line.split(",")]
            data = dict(zip(field_names, data_fields))
            p = Measurement(data)
            current_scan.measurements.append(p)

    return scans


def plot(input_file, current="digital"):

    scans = get_scans_from_file(input_file)

    fig, ax = plt.subplots(1, 1)

    ##
    ## configure the "pad"
    ##
    ax.tick_params(
        axis="both",
        which="both",
        labelsize=10,
        direction="in",
        labelleft=True,
        bottom=True,
        top=True,
        right=True,
        left=True,
    )

    x_data, y_data, z_data = [], [], []
    for scan in scans:
        for imeas, meas in enumerate(scan.measurements):

            x_data.append(meas.get_field("VDDA"))
            y_data.append(meas.get_field("VDDD"))
            if current.lower() == "digital":
                z_data.append(meas.get_field("PS_I_D"))
            elif current.lower() == "analog":
                z_data.append(meas.get_field("PS_I_A"))

    X = np.linspace(min(x_data), max(x_data), 51)
    Y = np.linspace(min(y_data), max(y_data), 51)
    X, Y = np.meshgrid(X, Y)
    Z = griddata((x_data, y_data), z_data, (X, Y), method="nearest")

    p = ax.pcolormesh(X, Y, Z, cmap="YlOrRd", shading="auto")

    ax.set_xlabel("SCC VDDA [V]", fontsize=12)
    ax.set_ylabel("SCC VDDD [V]", fontsize=12)

    cb = fig.colorbar(p)
    if current == "digital":
        z_label = "Digital Current [A]"
    elif current == "analog":
        z_label = "Analog Current [A]"
    cb.set_label(z_label, fontsize=12)

    ##
    ## other labels, etc
    ##
    text = "$\\bf{RD53b}$"
    text += f":{scans[0].chip_id}"
    size = 12
    opts = dict(transform=ax.transAxes)
    opts.update(dict(va="top", ha="left"))
    ax.text(0.0, 1.04, text, size=size, **opts)

    ## pause for realtime viewing
    fig.show()
    input()

    return True, ""


def plot_summary(input_file, current="digital"):

    total_scans = get_scans_from_file(input_file)
    ramp_up_scans, ramp_down_scans = [], []

    for s in total_scans:
        if "ramp" in s.header:
            if s.header["ramp"].lower() == "up":
                ramp_up_scans.append(s)
            elif s.header["ramp"].lower() == "down":
                ramp_down_scans.append(s)

    if len(total_scans) == 0:
        return False, "No scans loaded!"

    fixed_ramp_quantity = ""
    for key, val in total_scans[0].header.items():
        if key == "VDDA" or key == "VDDD":
            fixed_ramp_quantity = key
    if fixed_ramp_quantity == "":
        return False, "Could not determine fixed ramp quantity!"

    for iscans, scans in enumerate([ramp_up_scans, ramp_down_scans]):
        ramp_directions = ["up", "down"]
        if not len(scans):
            print(
                f"WARNING Scan list for ramp direction {ramp_directions[iscans].upper()} is empty"
            )
            continue

        fig, ax = plt.subplots(1, 1)

        ##
        ## ticks and stuff
        ##
        ax.tick_params(
            axis="both",
            which="both",
            labelsize=10,
            direction="in",
            labelleft=True,
            bottom=True,
            top=True,
            right=True,
            left=True,
        )

        ##
        ## get the data
        ##
        x_data, y_data, z_data = [], [], []
        for scan in scans:
            for imeasurement, measurement in enumerate(scan.measurements):
                if fixed_ramp_quantity == "VDDA":
                    x_data.append(measurement.get_field("VDDA"))
                    y_data.append(measurement.get_field("VDDD"))
                elif fixed_ramp_quantity == "VDDD":
                    x_data.append(measurement.get_field("VDDD"))
                    y_data.append(measurement.get_field("VDDA"))

                if current.lower() == "digital":
                    z_data.append(measurement.get_field("PS_I_D"))
                else:
                    z_data.append(measurement.get_field("PS_I_A"))

        X = np.linspace(min(x_data), max(x_data), 60)
        Y = np.linspace(min(y_data), max(y_data), 60)
        X, Y = np.meshgrid(X, Y)
        Z = griddata((x_data, y_data), z_data, (X, Y), method="nearest")

        ##
        ## draw
        ##
        p = ax.pcolormesh(X, Y, Z, cmap="YlOrRd", shading="auto")
        cb = fig.colorbar(p)

        ##
        ## labels
        ##
        labels = {
            True: ["SCC VDDA [V]", "SCC VDDD [V]"],
            False: ["SCC VDDD [V]", "SCC VDDA [V]"],
        }[fixed_ramp_quantity == "VDDA"]
        ax.set_xlabel(labels[0], fontsize=12)
        ax.set_ylabel(labels[1], fontsize=12)
        zlabel = {True: "Analog Current [A]", False: "Digital Current [A]"}[
            current == "analog"
        ]
        cb.set_label(zlabel, fontsize=12)

        ## team logos
        text = "$\\bf{RD53b}$"
        text += f":{scans[0].chip_id}, Ramp:{ramp_directions[iscans]}, Fixed:{fixed_ramp_quantity}"
        size = 12
        opts = dict(transform=ax.transAxes)
        opts.update(dict(va="top", ha="left"))
        ax.text(0.0, 1.05, text, size=size, **opts)

        ##
        ## pause
        ##
        fig.show()
        input()

    return True, ""
