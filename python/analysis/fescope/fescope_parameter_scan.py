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


def plot_feshape_for_parameters(data_param_map, param_name = "", parameter_values = [], do_error = False) :

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
            with open(toa_filename, "r") as toa_file, open(tot_filename, "r") as tot_file :
                all_toa_data = np.array(json.load(toa_file)["Data"])
                all_tot_data = np.array(json.load(tot_file)["Data"])

                ##
                ## remove LR columns
                ##
                all_toa_data = all_toa_data[2:398,:]
                all_tot_data = all_tot_data[2:398,:]

                ##
                ## select nonzero data (i.e. select only those pixels with 100% occupancy)
                ##
                idx = all_toa_data > 0
                all_toa_data = all_toa_data[idx]
                all_tot_data = all_tot_data[idx]


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
                if n_nonzero_pixels > 0 :
                    if n <= 0.99 * n_nonzero_pixels :
                        break
                n_nonzero_pixels = n

                ##
                ## subtract off the per-pixel average PToT mean when running a digital scan
                ##
                ptoa_digital_scan = 190.5 # TODO: fix to be per-pixel (need to check implementation of PToT digital scan)
                all_toa_data = all_toa_data - ptoa_digital_scan

                ##
                ## rising edge points of interest and convert to ns
                ##
                mean_rising_edge = np.mean(all_toa_data) * 1.5625
                stddev_rising_edge = np.std(all_toa_data) * 1.5625
                x_vals_rising_edge.append(mean_rising_edge)
                x_err_rising_edge.append(stddev_rising_edge)
                y_vals_rising_edge.append(th)

                ##
                ## falling edge points of interest and convert to ns
                ##
                mean_falling_edge = np.mean(all_toa_data + all_tot_data) * 1.5625
                stddev_falling_edge = np.std(all_toa_data + all_tot_data) * 1.5625
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


        ax.plot(x_vals_rising_edge, y_vals_rising_edge, color = param_colors[iparam], label = f"{param_name}-{parameter_val}")
        ax.plot(x_vals_falling_edge, y_vals_falling_edge, color = param_colors[iparam])

        ##
        ## draw the standard deviation in time on the falling edge of the pulse shape
        ## as a filled in area
        ##
        if do_error :
            ax.fill_betweenx(y_vals_falling_edge, x_vals_falling_edge - stddev_falling_edge, x_vals_falling_edge + stddev_falling_edge, alpha = 0.3, color = param_colors[iparam])

    max_x = 500
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
    args = parser.parse_args()


    all_dirs = glob.glob(f"{args.input}/*")
    param_vals = []
    param_name = ""
    data_param_map = {}
    for i, d in enumerate(all_dirs) :
        if i == 0 :
            param_name = d.split("/")[-1].split("_")[0]
        val = int(d.split("/")[-1].split("_")[-1])
        param_vals.append(val)
        complete_path = f"{d}/last_scan"
        data_param_map[val] = get_thresholds_from_input(complete_path)

    if args.name != "" :
        param_name = args.name

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

    plot_feshape_for_parameters(data_param_map, param_name, param_vals, args.error)

    

if __name__ == "__main__" :
    main()
