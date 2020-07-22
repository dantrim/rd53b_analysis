import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata

import sys


class Measurement:
    def __init__(self):

        self.vdda = 0.0
        self.vddd = 0.0
        self.ps_vdda = 0.0
        self.ps_vddd = 0.0
        self.analog_current = 0.0
        self.digital_current = 0.0


class Scan:
    def __init__(self, vdda):
        self.vdda = vdda
        self.measurements = []


def get_scans_from_file(input_file):

    scan_data = {}

    vdda_current = -1
    scan_lines = []
    field_names = []
    with open(input_file, "r") as infile:
        for iline, line in enumerate(infile):
            line = line.strip()
            if "RD53B" in line:

                if vdda_current > 0:
                    scan_data[vdda_current] = scan_lines
                    scan_lines = []

                vdda_current = float(line.split(":")[-1])
                continue

            if "I_lim" in line:
                field_names = line.split()
                continue
            scan_lines.append(line)

    field_map = {}
    for i, field in enumerate(field_names):
        field_map[field] = i

    scans = []
    for vdda in scan_data:
        s = Scan(float(vdda))
        scan_lines = scan_data[vdda]
        for iline, line in enumerate(scan_lines):
            data_fields = line.split()

            p = Measurement()
            p.vdda = float(data_fields[field_map["VDDA"]])
            p.vddd = float(data_fields[field_map["VDDD"]])
            p.ps_vddd = float(data_fields[field_map["PS_VD_set"]])
            p.ps_vdda = float(data_fields[field_map["PS_VA_set"]])
            p.analog_current = float(data_fields[field_map["PS_I_A"]])
            p.digital_current = float(data_fields[field_map["PS_I_D"]])
            s.measurements.append(p)
        scans.append(s)

    scans.sort(key=lambda x: x.vdda)
    return scans


def plot(input_file, current="digital"):

    scans = get_scans_from_file(input_file)

    fig, ax = plt.subplots(1, 1)

    x_data, y_data, z_data = [], [], []
    for scan in scans:
        for imeas, meas in enumerate(scan.measurements):

            x_data.append(meas.vdda)
            y_data.append(meas.vddd)
            if current.lower() == "digital":
                z_data.append(meas.digital_current)
            elif current.lower() == "analog":
                z_data.append(meas.analog_current)

    X = np.linspace(min(x_data), max(x_data), 51)
    Y = np.linspace(min(y_data), max(y_data), 51)
    X, Y = np.meshgrid(X, Y)
    Z = griddata((x_data, y_data), z_data, (X, Y), method="nearest")

    p = ax.pcolormesh(X, Y, Z, cmap="YlOrRd", shading="auto")

    ax.set_xlabel("SCC VDDA [V]")
    ax.set_ylabel("SCC VDDD [V]")

    cb = fig.colorbar(p)
    if current == "digital":
        z_label = "Digital Current [A]"
    elif current == "analog":
        z_label = "Analog Current [A]"
    cb.set_label(z_label)
    fig.show()
    x = input()

    return True, str(x)
