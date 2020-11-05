#!/bin/env python

from pathlib import Path
import sys
from argparse import ArgumentParser
from scipy.interpolate import griddata
from scipy.stats import norm
from scipy.optimize import curve_fit

import json
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import glob

from plot_fescope_json import get_thresholds_from_input

def dac_to_mv_conversion(dac_data_file) :

    if dac_data_file == "" :
        return {}

    dac2mv = {}
    with open(dac_data_file, "r") as infile :
        for line in infile :
            line = line.strip()
            if line.startswith("#") : continue
            fields = [x.strip() for x in line.split(",")]
            dac_setting = int(fields[0])
            voltage = float(fields[1])
            voltage_mV = voltage * 1e3
            dac2mv[dac_setting] = voltage_mV
    return dac2mv

def get_sim_data(simulation_data_file) :

    x_vals, y_vals = [], []
    if simulation_data_file == "" :
        return [], []

    with open(simulation_data_file, "r") as infile :
        for line in infile :
            line = line.strip()
            if line.startswith("#") : continue
            fields = line.split()
            time = float(fields[0])
            time_ns = 1e9 * time
            pulse = float(fields[2]) # get the data at 3000e (field[1] is 1500e)
            pulse_mV = pulse * 1e3

            x_vals.append(time_ns)
            y_vals.append(pulse_mV)

    ##
    ## remove a.u. offset
    ##
    voffset = y_vals[0]
    tmp_y_vals = []
    for val in y_vals :
        if voffset < 0 :
            tmp_y_vals.append( val + abs(voffset) )
        else :
            tmp_y_vals.append( val - abs(voffset) )
    y_vals = tmp_y_vals

    ##
    ## multiplier
    ##
    peak = max(y_vals)
    multiplier = 1
    tmp_y_vals = []
    for val in y_vals :
        tmp_y_vals.append( val * multiplier )
    y_vals = tmp_y_vals

    return x_vals, y_vals

def core_generator(toa_map, tot_map, digital_offset_map) :

    n_core_columns = 50
    n_core_rows = 48
    
    core_col_edges = []
    core_row_edges = []
    core_edges = []
    
    col_low = 0
    row_low = 0
    for i in range(n_core_columns) :
        low_edge = col_low
        if col_low <= 1 :
            low_edge = 2
        high_edge = col_low + 8
        if high_edge >= 398 :
            high_edge = 397
        edge = (low_edge, high_edge)
        col_low += 8
        core_col_edges.append(edge)
    for i in range(n_core_rows) :
        edge = (row_low, row_low + 8)
        row_low += 8
        core_row_edges.append(edge)
    
    core_num = 0
    for icol, col_edges in enumerate(core_col_edges) :
        for irow, row_edges in enumerate(core_row_edges) :
            pix_columns = np.arange(col_edges[0], col_edges[1], 1)
            pix_rows = np.arange(row_edges[0], row_edges[1], 1)
    
            c = (col_edges[0], col_edges[1])
            r = (row_edges[0], row_edges[1])
            #print(f"Core Number {core_num}: ({c}, {r})")
            core_num+= 1
            yield toa_map[c[0]:c[1], r[0]:r[1]], tot_map[c[0]:c[1], r[0]:r[1]], digital_offset_map[c[0]:c[1], r[0]:r[1]]
    
        #yield toa_map[col_edges[0]:col_edges[1], row_edges[0]:row_edges[1]], tot_map[col_edges[0]:col_edges[1], row_edges[0]:row_edges[1]], digital_offset_map[col_edges[0]:col_edges[1], row_edges[0]:row_edges[1]]

def deleteFrom2D(arr2D, row, column):
    'Delete element from 2D numpy array by row and column position'
    modArr = np.delete(arr2D, row * arr2D.shape[1] + column)
    return modArr

def plot_feshape_overlay(data_map, param_name = "",
        parameter_values = [], do_error = False, digital_scan_toa_map_file = None,
        select_pixel_address = (-1,-1), threshold_truncate = -1,
        subtract_vertical_offset = False, sim_data_file = "", dac_data_file = "",
        do_core_averaging = False) :

    sim_x_vals, sim_y_vals = get_sim_data(sim_data_file)
    dac2mv_map = dac_to_mv_conversion(dac_data_file)

    max_x = -1
    max_y = -1

    ##
    ## create the pad for plotting
    ##
    fig, ax = plt.subplots(1,1)
    ax.set_xlabel(f"Time [ns]")
    ax.set_ylabel(r"$\Delta$th [counts]")
    if dac2mv_map :
        ax.set_ylabel(r"$\Delta$th [mV]")
    ax.tick_params(which = "both", direction = "in", top = True, bottom = True, left = True, right = True)

    rising_edge_fix_map = {}

    thresholds = sorted(data_map.keys())


    n_nonzero_pixels = 0

    t_offset = -1
    n_cores_max = -1
    core_map_z = {}

    core_toa_vals = {}
    core_tot_vals = {}

    for ith, th in enumerate(thresholds) :
        n_core_toa = 0

        if th > 220 :
            print(f"Breaking early (at th = {th}) from th scan!")
            break

        x_vals_rising_edge, y_vals_rising_edge = np.array([]), np.array([])
        x_vals_falling_edge, y_vals_falling_edge = np.array([]), np.array([])

        n_cores = 0
        core_toa_vals[th] = []
        core_tot_vals[th] = []

        tot_filename, toa_filename = data_map[th]
        with open(toa_filename, "r") as toa_file, open(tot_filename, "r") as tot_file, open(digital_scan_toa_map_file, "r") as digital_toa_file :
            all_toa_data = np.array(json.load(toa_file)["Data"])
            all_tot_data = np.array(json.load(tot_file)["Data"])
            digital_offset_map = np.array(json.load(digital_toa_file)["Data"])

            t_conv = 1.5625
            #if do_core_averaging :
            if True :

                pulse_height = th

                for core_toa, core_tot, core_digital in core_generator(all_toa_data, all_tot_data, digital_offset_map) :

                    #if not do_core_averaging :
                    #    core_toa = core_toa[2:398,:]
                    #    core_tot = core_tot[2:398,:]
                    #    core_digital = core_digital[2:398,:]

                    #if core_toa.shape[0] != 8 :
                    #    continue

                    idx = (core_toa > 0)# & (core_tot > 0) & (core_digital > 0)
                    if not np.any(idx) :
                        continue

                    core_is_bad = False
                    if ith == 11 and core_toa.shape[0] == 8 : #and n_cores < 100 :
                        fills = 0
                        for col in np.arange(0,8,1) :
                            for row in np.arange(0,8,1) :
                                address = (col,row)
                                if address not in core_map_z :
                                    core_map_z[address] = []
                                toa = core_toa[col][row] - core_digital[col][row]
                                toa *= t_conv
                                toa -= 12
                                if toa < 0 : continue
                                core_map_z[address].append(toa)
                    n_cores += 1

                    core_toa = core_toa[idx]
                    core_tot = core_tot[idx]
                    core_digital = core_digital[idx]

                    core_toa = core_toa - core_digital
                    pos_idx = core_toa > 0
                    core_toa = core_toa[pos_idx]
                    core_tot = core_tot[pos_idx]

                    core_toa *= t_conv
                    core_tot *= t_conv

                    n_core_toa += core_toa.size

                    avg_core_toa = np.mean(core_toa)
                    avg_core_tot = np.mean(core_tot)

                    if do_core_averaging :
                        core_toa_vals[th].append(avg_core_toa)
                    else :
                        core_toa_vals[th] += list(core_toa)
                    #core_tot_vals[th].append(avg_core_tot)

                    ##
                    ## convert to mV if loaded a DAC-to-mV file
                    ##
                    if dac2mv_map :
                        pulse_height = dac2mv_map[th]

                    if do_core_averaging :
                        x_vals_rising_edge = np.concatenate([x_vals_rising_edge, [avg_core_toa]])
                        x_vals_falling_edge = np.concatenate([x_vals_falling_edge, [avg_core_toa + avg_core_tot]])
                        y_vals_rising_edge = np.concatenate([y_vals_rising_edge, [pulse_height]])
                        y_vals_falling_edge = np.concatenate([y_vals_falling_edge, [pulse_height]])
                    else :
                        x_vals_rising_edge = np.concatenate([x_vals_rising_edge, core_toa])
                        falling_edge = core_toa + core_tot
                        x_vals_falling_edge = np.concatenate([x_vals_falling_edge, falling_edge])
                        y_vals_rising_edge = np.concatenate([y_vals_rising_edge, np.ones(len(core_toa)) * pulse_height])
                        y_vals_falling_edge = np.concatenate([y_vals_falling_edge, np.ones(len(falling_edge)) * pulse_height])

                if t_offset < 0 :
                    arr = x_vals_rising_edge[~np.isnan(x_vals_rising_edge)]
                    offset = np.mean(arr)
                    std = np.std(arr)
                    if not np.isnan(offset) :
                        t_offset = offset - std
                t_offset = 12

                print(f"Threshold = {th} counts: Number of cores considered: {n_cores}")
                n_cores_max = max([n_cores, n_cores_max])

            ##
            ## boundaries for plotting
            ##
            max_x = max([max_x, np.max(x_vals_falling_edge)])

            y_to_check = th
            #max_y = max([max_y, np.max(y_vals_falling_edge)])
            max_y = 500

            x = np.concatenate([x_vals_rising_edge, x_vals_falling_edge])
            x -= (t_offset + 4)
            y = np.concatenate([y_vals_rising_edge, y_vals_falling_edge])
            x_bw = 1
            y_bw = dac2mv_map[10]
            max_x = 200
            x_bins = np.arange(0, max_x + x_bw, x_bw)
            y_bins = np.arange(0, max_y + y_bw, y_bw)
            cmin = 10
            if not do_core_averaging :
                cmin = 100
            h, xedges, yedges, im = ax.hist2d(x,y, bins = (x_bins, y_bins), norm = matplotlib.colors.LogNorm(), cmap = plt.cm.YlOrRd, cmin = 10)

        if ith == 11 : #and do_core_averaging :

            arr = x_vals_rising_edge - (t_offset + 4)
            h, edges = np.histogram(arr, bins = np.arange(0, max_x, 1.5625))
            mean, std = np.mean(arr), np.std(arr)
            arr_fit = arr#[(arr > (mean-std)) & (arr < (mean+std))]
            mu, sigma = norm.fit(arr_fit)
            print(f"THRESHOLD = {dac2mv_map[th]} mV: ToA: {mu} +/- {sigma} ns")

    #if subtract_vertical_offset :
    #    y_vals_rising_edge = [x - y_vals_rising_edge[0] for x in y_vals_rising_edge]
    #    y_vals_falling_edge = [x - y_vals_falling_edge[0] for x in y_vals_falling_edge]

    if threshold_truncate >= 0 :
        idx = (y_vals_rising_edge <= threshold_truncate) & (y_vals_falling_edge <= threshold_truncate)

        x_vals_rising_edge = x_vals_rising_edge[idx]
        y_vals_rising_edge = y_vals_rising_edge[idx]
        x_vals_falling_edge = x_vals_falling_edge[idx]
        y_vals_falling_edge = y_vals_falling_edge[idx]

    #x = np.concatenate([x_vals_rising_edge, x_vals_falling_edge])
    #y = np.concatenate([y_vals_rising_edge, y_vals_falling_edge])

    ##
    ## apply ∆t offset
    ##
    print(f"Applying ∆t offset of {t_offset} ns")
    #x -= (t_offset - 2)
    #x -= (t_offset + 2)

#    x_bw = 1.5625
#    y_bw = 10
#    max_x = 200
#    x_bins = np.arange(0, max_x + x_bw, x_bw)
#    y_bins = np.arange(0, max_y + y_bw, y_bw)
#    h, xedges, yedges, im = ax.hist2d(x,y, bins = (x_bins, y_bins), norm = matplotlib.colors.LogNorm(), cmap = plt.cm.YlOrRd)

    #max_x = 50
    max_y = 200
    max_x = 200
    ax.set_xlim([0, max_x])
    ax.set_ylim([0, 1.1 * max_y])
    ax.plot([0, max_x], [88.4, 88.4], "k-")

    x_ticks = np.arange(0, max_x + 25, 25)
    ax.set_xticks(x_ticks)
    x_tick_labels = []
    for tick in x_ticks :
        if tick % 100 == 0 :
            x_tick_labels.append(f"{tick}")
        else :
            x_tick_labels.append("")
    ax.set_xticklabels(x_tick_labels)
    ax.grid(which = "both")


    ##
    ## metadata text
    ##
    ax.text(0.02, 1.02, "RD53b Frontend Scope", transform = ax.transAxes, weight = "bold")

    if select_pixel_address[0] >= 0 or select_pixel_address[1] >= 0 :
        ax.text(0.39, 1.02, f": Pixel ({select_pixel_address[0]},{select_pixel_address[1]})", transform = ax.transAxes, weight = "bold")
    elif do_core_averaging :
        ax.text(0.39, 1.02, f": {n_cores_max} cores (per-core averaging)", transform = ax.transAxes, weight = "bold")
    else :
        ax.text(0.39, 1.02, f": Many pixels", transform = ax.transAxes, weight = "bold")

    ##
    ## simulation data overlay
    ##
    if len(sim_x_vals) > 0 :
        sim_x_plot, sim_y_plot = [], []
        for ival, val in enumerate(sim_x_vals) :
            if val >= 0 and val < max_x :
                sim_x_plot.append(val)
                sim_y_plot.append(sim_y_vals[ival])
        ax.plot(sim_x_plot, sim_y_plot, "k-", label = "Simulation")


    ##
    ## legend
    ##
    ax.legend(loc = "best")

    fig.show()
    x = input()
    fig.savefig("fescope_plot.pdf", bbox_inches = "tight")

    #if do_core_averaging :
    if True :
        x_data = []
        y_data = []
        z_data = []
        mean0 = 0
        for col in np.arange(0,8,1) :
            for row in np.arange(0,8,1) :
                address = (col,row)
                z = np.array(core_map_z[address])
                mean = np.mean(z)
                std = np.std(z)
                if col == 0 and row == 0 :
                    mean0 = mean
                delta = mean - mean0 
                x_data.append(col)
                y_data.append(row)
                z_data.append(std)
        print(f"MEAN Z_DATA = {np.mean(z_data)} ns")
                
        X = np.linspace(min(x_data), max(x_data), 8)
        Y = np.linspace(min(y_data), max(y_data), 8)
        X, Y = np.meshgrid(X,Y)
        Z = griddata((x_data, y_data), z_data, (X,Y), method = "nearest")

        fig, ax = plt.subplots(1,1)
        p = ax.pcolormesh(X,Y,Z, cmap = "YlOrRd", shading = "auto")
        cb = fig.colorbar(p)
        cb.set_label(f"Std. Dev. of PToA Over Core Pixels ({n_cores_max} cores)")
        fig.show()
        _ = input()
        

def main() :

    parser = ArgumentParser()
    parser.add_argument("-i", "--input", required = True,
        help = "Path to directory containing data from parameter scans"
    )
    parser.add_argument("-v", "--value", default = "",
        help = "Select a specific value"
    )
    parser.add_argument("-l", "--list", action = "store_true", default = False,
        help = "List the parameter values found in the input directory"
    )
    parser.add_argument("-n", "--name", default = "",
        help = "Provide a nice name for the parameter"
    )
    parser.add_argument("-e", "--error", default = False, action = "store_true",
        help = "Show std deviation of falling edge"
    )
    parser.add_argument("-d", "--digital-scan", default = "",
        help = "Provide a DigitalScan ToA map for defining the per-pixel subtraction"
    )
    parser.add_argument("-p", "--pixel", default = "-1:-1",
        help = "Provide pixel address in format <col>:<row>"
    )
    parser.add_argument("-t", "--truncate", default = -1,
        help = "Truncate the thresholds to be below the specified value"
    )
    parser.add_argument("--voff", default = False, action = "store_true",
        help = "Subtract any vertical offset in FE pulse"
    )
    parser.add_argument("--sim-data", default = "",
        help = "Provide data from simulation for comparison"
    )
    parser.add_argument("--dac-data", default = "",
        help = "Provide a file with DAC-to-mV relationship"
    )
    parser.add_argument("--core-avg", default = False, action = "store_true")
    args = parser.parse_args()

    all_dirs = glob.glob(f"{args.input}/*")
    param_vals = []
    param_name = ""
    data_map = get_thresholds_from_input(args.input)

    args.truncate = int(args.truncate)

    select_pixel_address = [int(x) for x in args.pixel.strip().split(":")]
    select_col, select_row = select_pixel_address
    select_pixel_address = (select_col, select_row)

    p_digital = Path(args.digital_scan)
    ok = p_digital.exists() and p_digital.is_file()
    if not ok :
        print(f"ERROR Provided digital scan ToA map could not be found: \"{args.digital_scan}\"")
        sys.exit(1)

    plot_feshape_overlay(data_map, param_name, param_vals, args.error, args.digital_scan, select_pixel_address, args.truncate, args.voff, args.sim_data, args.dac_data, args.core_avg)

    

if __name__ == "__main__" :
    main()
