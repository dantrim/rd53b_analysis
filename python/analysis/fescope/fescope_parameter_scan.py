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


def plot_feshape_for_parameters(data_param_map, param_name = "",
        parameter_values = [], do_error = False, digital_scan_toa_map_file = None,
        select_pixel_address = (-1,-1)) :

    #parameter_values = sorted(data_param_map.keys())

    max_x = -1
    max_y = -1

    ##
    ## create the pad for plotting
    ##
    fig, ax = plt.subplots(1,1)
    ax.set_xlabel(f"Time [ns]")
    ax.set_ylabel(r"$\Delta$th [counts]")
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
                    if n <= 0.99 * n_nonzero_pixels :
                        break
                n_nonzero_pixels = n

                ##
                ## subtract off the per-pixel average PToT mean when running a digital scan
                ##
                all_toa_data = all_toa_data - digital_offset_map
                #else :
                #    ptoa_digital_scan = 190.5
                #    all_toa_data = all_toa_data - ptoa_digital_scan

                ##
                ## rising edge points of interest and convert to ns
                ##
                t_conv = 1.0
                mean_rising_edge = np.mean(all_toa_data) * t_conv

                ##
                ## clean out unfilled data
                ##
                if np.isnan(mean_rising_edge) :
                    continue

                if mean_rising_edge < 0 :
                    print(f"[param={parameter_val}] [th={th}] Mean rising edge NEGATIVE! {mean_rising_edge}")
                    continue

                stddev_rising_edge = np.std(all_toa_data) * t_conv 
                x_vals_rising_edge.append(mean_rising_edge)
                x_err_rising_edge.append(stddev_rising_edge)
                y_vals_rising_edge.append(th)

                ##
                ## falling edge points of interest and convert to ns
                ##
                mean_falling_edge = np.mean(all_toa_data + all_tot_data) * t_conv
                stddev_falling_edge = np.std(all_toa_data + all_tot_data) * t_conv
                x_vals_falling_edge.append(mean_falling_edge)
                x_err_falling_edge.append(stddev_falling_edge)
                y_vals_falling_edge.append(th)

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

        y_vals_rising_edge = [x - y_vals_rising_edge[0] for x in y_vals_rising_edge]
        y_vals_falling_edge = [x - y_vals_falling_edge[0] for x in y_vals_falling_edge]
        ax.plot(x_vals_rising_edge, y_vals_rising_edge, color = param_colors[iparam], label = f"{param_name}:{parameter_label}")
        ax.plot(x_vals_falling_edge, y_vals_falling_edge, color = param_colors[iparam])

        ##
        ## draw the standard deviation in time on the falling edge of the pulse shape
        ## as a filled in area
        ##
        if do_error :
            ax.fill_betweenx(y_vals_falling_edge, x_vals_falling_edge - stddev_falling_edge, x_vals_falling_edge + stddev_falling_edge, alpha = 0.3, color = param_colors[iparam])

    max_x = 250
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
    ax.legend(loc = "best")

    ##
    ## metadata text
    ##
    ax.text(0.02, 1.02, "RD53b Frontend Scope", transform = ax.transAxes, weight = "bold")

    if select_pixel_address[0] >= 0 or select_pixel_address[1] >= 0 :
        ax.text(0.39, 1.02, f": Pixel ({select_pixel_address[0]},{select_pixel_address[1]})", transform = ax.transAxes, weight = "bold")
    else :
        ax.text(0.39, 1.02, f": Many pixels", transform = ax.transAxes, weight = "bold")

    fig.show()
    x = input()

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

    plot_feshape_for_parameters(data_param_map, param_name, param_vals, args.error, args.digital_scan, select_pixel_address)

    

if __name__ == "__main__" :
    main()
