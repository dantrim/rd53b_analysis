#!/bin/env python

from pathlib import Path
import sys
from argparse import ArgumentParser

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

def plot_feshape_for_parameters(data_param_map, param_name = "",
        parameter_values = [], do_error = False, digital_scan_toa_map_file = None,
        select_pixel_address = (-1,-1), threshold_truncate = -1,
        subtract_vertical_offset = False, sim_data_file = "", dac_data_file = "") :

    #parameter_values = sorted(data_param_map.keys())

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

    ##
    ## set the color cycle so that each parameter has a unique color
    ##
    param_colors = []
    n_colors = len(parameter_values)
    cm = plt.get_cmap("gist_rainbow")
    for iparam in range(n_colors) :
        param_colors.append( cm(1.0 * iparam / n_colors) )

    

    for iparam, parameter_val in enumerate(parameter_values) :

        rising_edge_fix_map = {}
        print(f"[{iparam+1:02d}/{len(parameter_values):02d}] Gathering data for {param_name} = {parameter_val}")

        data_map = data_param_map[parameter_val]
        thresholds = sorted(data_map.keys())

        x_vals_rising_edge, x_err_rising_edge = [], []
        x_vals_falling_edge, x_err_falling_edge = [], []
        y_vals_rising_edge, y_vals_falling_edge = [], []
 
        x_err_left_falling_edge, x_err_right_falling_edge = [], []

        n_nonzero_pixels = 0

        for ith, th in enumerate(thresholds) :
            tot_filename, toa_filename = data_map[th]
            with open(toa_filename, "r") as toa_file, open(tot_filename, "r") as tot_file, open(digital_scan_toa_map_file, "r") as digital_toa_file :
                all_toa_data = np.array(json.load(toa_file)["Data"])
                all_tot_data = np.array(json.load(tot_file)["Data"])
                digital_offset_map = np.array(json.load(digital_toa_file)["Data"])


                ##
                ## remove LR columns or specific pixel selected by the user
                ##
                select_col, select_row = select_pixel_address
                if select_col >= 0 or select_row >= 0 :
                    if select_row < 0 and select_col >= 0:
                        all_toa_data =             all_toa_data[select_col,:]
                        all_tot_data =             all_tot_data[select_col,:]
                        digital_offset_map = digital_offset_map[select_col,:]
                    elif select_col < 0 and select_row >= 0 :
                        all_toa_data =             all_toa_data[:,select_row]
                        all_tot_data =             all_tot_data[:,select_row]
                        digital_offset_map = digital_offset_map[:,select_row]
                    else :
                        all_toa_data =             all_toa_data[select_col,select_row]
                        all_tot_data =             all_tot_data[select_col,select_row]
                        digital_offset_map = digital_offset_map[select_col,select_row]
                else :
                    all_toa_data = all_toa_data[2:398,:]
                    all_tot_data = all_tot_data[2:398,:]
                    digital_offset_map = digital_offset_map[2:398,:]


                ##
                ## select nonzero data (i.e. select only those pixels with 100% occupancy)
                ##
                idx = (all_toa_data > 0) & (all_tot_data > 0) & (digital_offset_map > 0)
                all_toa_data = all_toa_data[idx]
                all_tot_data = all_tot_data[idx]
                digital_offset_map = digital_offset_map[idx]


                ##
                ## set the rising edge to have all the same mean at each height (threshold)
                ##
                if th not in rising_edge_fix_map :
                    rising_edge_fix_map[th] = np.mean(all_toa_data)
                all_toa_data = all_toa_data + (rising_edge_fix_map[th] - all_toa_data)

                ##
                ## check if we have peaked or not, if so don't include those points
                ##
                n = len(all_toa_data)
                #if th >= 500 :
                #    break
                if n_nonzero_pixels > 0 :
                    if n <= 0.8 * n_nonzero_pixels :
                        break
                n_nonzero_pixels = n

                ##
                ## subtract off the per-pixel average PToT mean when running a digital scan
                ##
                all_toa_data = all_toa_data - digital_offset_map

                ##
                ## rising edge points of interest and convert to ns
                ##
                t_conv = 1.5625
                mean_rising_edge = np.mean(all_toa_data) * t_conv

                ##
                ## clean out unfilled data
                ##
                if np.isnan(mean_rising_edge) :
                    continue

                if mean_rising_edge < 0 :
                    print(f"[param={parameter_val}] [th={th}] Mean rising edge NEGATIVE! {mean_rising_edge}")
                    continue

                ##
                ## convert to mV if loaded a DAC-to-mV file
                ##
                if dac2mv_map :
                    th = dac2mv_map[th]

                stddev_rising_edge = np.std(all_toa_data) * t_conv 
                x_vals_rising_edge.append(mean_rising_edge)
                x_err_rising_edge.append(stddev_rising_edge)
                y_vals_rising_edge.append(th)

                ##
                ## falling edge points of interest and convert to ns
                ##
                falling_edge_data = (all_toa_data + all_tot_data) * t_conv
                left_edge_falling_edge = np.min(falling_edge_data)
                right_edge_falling_edge = np.max(falling_edge_data)
                x_err_left_falling_edge.append(left_edge_falling_edge)
                x_err_right_falling_edge.append(right_edge_falling_edge)

                mean_falling_edge = np.mean(all_toa_data + all_tot_data) * t_conv
                stddev_falling_edge = np.std(all_toa_data + all_tot_data) * t_conv
                x_vals_falling_edge.append(mean_falling_edge)
                x_err_falling_edge.append(stddev_falling_edge)
                y_vals_falling_edge.append(th)

                if th == 50 :
                    print(f"XXX Param: {parameter_val} Mean ToA at th = {th}: {mean_rising_edge} +/- {stddev_rising_edge}")

                ##
                ## boundaries for plotting
                ##
                x_to_check = mean_falling_edge + stddev_falling_edge
                max_x = max([max_x, x_to_check])

                y_to_check = th
                max_y = max([max_y, y_to_check])


        parameter_label = str(int(abs(parameter_val)))
        if parameter_val < 0 :
            parameter_label = "neg. " + str(parameter_label)


        
        print(f" *** WARNING: REMOVING DATA POINTS *** ")
        print(f" *** WARNING: REMOVING DATA POINTS *** ")
        print(f" *** WARNING: REMOVING DATA POINTS *** ")
        x_rising_tmp, x_falling_tmp = [], []
        y_rising_tmp, y_falling_tmp = [], []
        for i, val in enumerate(x_vals_rising_edge) :
            if i == 0 : continue
            x_rising_tmp.append(val)
            x_falling_tmp.append(x_vals_falling_edge[i])
            y_rising_tmp.append(y_vals_rising_edge[i])
            y_falling_tmp.append(y_vals_falling_edge[i])
        x_vals_rising_edge = x_rising_tmp
        x_vals_falling_edge = x_falling_tmp
        y_vals_rising_edge = y_rising_tmp
        y_vals_falling_edge = y_falling_tmp

        if subtract_vertical_offset :
            y_vals_rising_edge = [x - y_vals_rising_edge[0] for x in y_vals_rising_edge]
            y_vals_falling_edge = [x - y_vals_falling_edge[0] for x in y_vals_falling_edge]

        if threshold_truncate >= 0 :
            tmp_x_rising, tmp_x_falling = [], []
            tmp_y_rising, tmp_y_falling = [], []
            for ival, val in enumerate(y_vals_rising_edge) :
                if val > threshold_truncate : continue
                tmp_y_rising.append(val)
                tmp_x_rising.append(x_vals_rising_edge[ival])
            for ival, val in enumerate(y_vals_falling_edge) :
                if val > threshold_truncate : continue
                tmp_y_falling.append(val)
                tmp_x_falling.append(x_vals_falling_edge[ival])

            x_vals_rising_edge = tmp_x_rising
            x_vals_falling_edge = tmp_x_falling
            y_vals_rising_edge = tmp_y_rising
            y_vals_falling_edge = tmp_y_falling

            ##
            ## scale
            ##
            #y_vals_rising_edge = [x/threshold_truncate for x in y_vals_rising_edge]
            #y_vals_falling_edge = [x/threshold_truncate for x in y_vals_falling_edge]

        if param_name == "VCalHigh" :
            param_name = "∆Vcal"
            parameter_label = int(parameter_label) - 200


        ##
        ## ∆t offset
        ##
        #t_offset = x_vals_rising_edge[0] # th = 0
        if len(sim_x_vals) > 0 :
            t_offset = x_vals_rising_edge[0]
            #t_offset -= 1.44
            print(f"Applying ∆t shift in observed data of {t_offset} ns")
            x_vals_rising_edge = [x - abs(t_offset) for x in x_vals_rising_edge]
            x_vals_falling_edge = [x - abs(t_offset) for x in x_vals_falling_edge]

        ax.plot(x_vals_rising_edge, y_vals_rising_edge, color = param_colors[iparam], label = f"{param_name}:{parameter_label}")
        ax.plot(x_vals_falling_edge, y_vals_falling_edge, color = param_colors[iparam])

        ##
        ## draw the standard deviation in time on the falling edge of the pulse shape
        ## as a filled in area
        ##
        if do_error :
            #ax.fill_betweenx(y_vals_falling_edge, x_err_left_falling_edge, x_err_right_falling_edge, alpha = 0.3, color = param_colors[iparam])
            ax.fill_betweenx(y_vals_falling_edge, x_vals_falling_edge - stddev_falling_edge, x_vals_falling_edge + stddev_falling_edge, alpha = 0.3, color = param_colors[iparam])

    #max_x = 50
    max_y = 450
    max_x = 300
    ax.set_xlim([0, max_x])
    ax.set_ylim([0, 1.1 * max_y])

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
        print(f"FOO max sim_y_vals = {max(sim_y_plot)}")
        ax.plot(sim_x_plot, sim_y_plot, "k-", label = "Simulation")

    ##
    ## legend
    ##
    ax.legend(loc = "best")

    fig.show()
    x = input()
    fig.savefig("fescope_plot.pdf", bbox_inches = "tight")

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
    args = parser.parse_args()

    all_dirs = glob.glob(f"{args.input}/*")
    param_vals = []
    param_name = ""
    data_param_map = {}
    for i, d in enumerate(all_dirs) :
        if i == 0 :
            param_name = d.split("/")[-1].split("_")[0]
        val = d.split("/")[-1].split("_")[-1]
        is_neg = "neg" in val or "m" in val
        is_pos = "pos" in val or "p" in val
        val = int(val.replace("neg","").replace("m","").replace("pos","").replace("p",""))
        if is_neg :
            val = -1.0 * val
        param_vals.append(val)
        complete_path = f"{d}/last_scan"
        data_param_map[val] = get_thresholds_from_input(complete_path)

    args.truncate = int(args.truncate)

    if args.name != "" :
        param_name = args.name

    select_pixel_address = [int(x) for x in args.pixel.strip().split(":")]
    select_col, select_row = select_pixel_address
    select_pixel_address = (select_col, select_row)

    param_vals = sorted(param_vals)
    select_values = []
    if args.value != "" :
        values = [int(x) for x in args.value.strip().split(",")]
        param_vals = values
        tmp = {}
        for p in param_vals :
            tmp[p] = data_param_map[p]
        data_param_map = tmp
    ##
    ## only want to list parameters and their values
    ##
    if args.list :
        print(f"Loaded data for parameter: {param_name}")
        for ip, val in enumerate(param_vals) :
            print(f" [{ip:02d}] {val} ({len(data_param_map[val])} files)")
        sys.exit(0)

    p_digital = Path(args.digital_scan)
    ok = p_digital.exists() and p_digital.is_file()
    if not ok :
        print(f"ERROR Provided digital scan ToA map could not be found: \"{args.digital_scan}\"")
        sys.exit(1)

    plot_feshape_for_parameters(data_param_map, param_name, param_vals, args.error, args.digital_scan, select_pixel_address, args.truncate, args.voff, args.sim_data, args.dac_data)

    

if __name__ == "__main__" :
    main()
