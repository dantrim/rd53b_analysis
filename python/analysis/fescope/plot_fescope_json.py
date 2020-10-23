#!/bin/env python

from pathlib import Path
import sys
from argparse import ArgumentParser
import json
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import glob

def get_thresholds_from_input(input_directory) :

    all_files_tot = glob.glob(f"{input_directory}/*MeanPToTMap*.json")
    all_files_toa = glob.glob(f"{input_directory}/*MeanPToAMap*.json")
    #print(f"Found {len(all_files_tot)} files")

    if len(all_files_tot) != len(all_files_toa) :
        print(f"ERROR: Number of ToT files is not the same as ToA files!")
        sys.exit(1)

    thresholds = {}
    for f in all_files_tot :
        threshold = int(f.split("/")[-1].split("-")[-1].replace(".json",""))
        if threshold not in thresholds :
            thresholds[threshold] = []
        thresholds[threshold].append(f)

    for f in all_files_toa :
        threshold = int(f.split("/")[-1].split("-")[-1].replace(".json",""))
        if threshold not in thresholds :
            thresholds[threshold] = []
        thresholds[threshold].append(f)

    for th, files in thresholds.items() :
        if len(files) != 2 :
            print(f"ERROR: Did not find 2 files for threshold {th}")
            sys.exit(1)
            
    return thresholds

def load_tdac_map(tdac_map_file) :

    with open(tdac_map_file, "r") as infile :
        return np.array(json.load(infile)["Data"])

def plot_pixel(data_map, selected_pix_col, selected_pix_row) :

    thresholds = sorted(data_map.keys())

    x_vals_rising_edge, y_vals_rising_edge = np.array([]), np.array([])
    x_vals_falling_edge, y_vals_falling_edge = np.array([]), np.array([])

    n_nonzero_pixels = 0

    nonzero_pixels = []

    for ith, th in enumerate(thresholds) :
        tot_filename, toa_filename = data_map[th]
        with open(toa_filename, "r") as toa_file, open(tot_filename, "r") as tot_file :
            #if selected_pix_col >= 0 : #@and selected_pix_row >= 0 :
            if selected_pix_row >= 0 : #@and selected_pix_row >= 0 :
                toa_data = np.array(json.load(toa_file)["Data"])[selected_pix_col][:] # exclude 2 left-most and right-most columns
                tot_data = np.array(json.load(tot_file)["Data"])[selected_pix_col][:] # exclude 2 left-most and right-most columns
                #toa_data = np.array(json.load(toa_file)["Data"])[selected_pix_col][selected_pix_row]
                #tot_data = np.array(json.load(tot_file)["Data"])[selected_pix_col][selected_pix_row]
            else :
                toa_data = np.array(json.load(toa_file)["Data"])[2:398, :] # exclude 2 left-most and right-most columns
                tot_data = np.array(json.load(tot_file)["Data"])[2:398, :] # exclude 2 left-most and right-most columns

        nonzero_col, nonzero_row = np.nonzero(toa_data)
        for col, row in zip(nonzero_col, nonzero_row) :
            address = (col,row)
            if address not in nonzero_pixels :
                nonzero_pixels.append(address)

        idx = toa_data != 0
        toa_data = toa_data[ idx ] #toa_data != 0 ]
        tot_data = tot_data[ idx ] #tot_data != 0 ]

        #idx = toa_data > 220
        #toa_data = toa_data[ idx ] #toa_data != 0 ]
        #tot_data = tot_data[ idx ] #tot_data != 0 ]

    
        n_nonzero_pixels = max([n_nonzero_pixels, len(toa_data)])

        x_vals_rising_edge = np.concatenate([x_vals_rising_edge, toa_data])
        x_vals_falling_edge = np.concatenate([x_vals_falling_edge, toa_data + tot_data])
        y_vals_rising_edge = np.concatenate([y_vals_rising_edge, np.ones(len(toa_data)) * th])
        y_vals_falling_edge = np.concatenate([y_vals_falling_edge, np.ones(len(tot_data)) * th])

        #x_vals_rising_edge += [x for x in toa_data[:,0]]
        #y_vals_rising_edge += [th for x in x_vals_rising_edge]
        #x_vals_falling_edge += [x for x in toa_data[:,0] + tot_data[:,0]]
        #y_vals_falling_edge += [th for x in x_vals_falling_edge]
        #x_vals_rising_edge.append(toa_data[selected_pix_col][selected_pix_row])
        #y_vals_rising_edge.append(th)
        #x_vals_falling_edge.append(toa_data[selected_pix_col][selected_pix_row] + tot_data[selected_pix_col][selected_pix_row])
        #y_vals_falling_edge.append(th)

    print(f"Loaded non-zero data from {n_nonzero_pixels} pixels!")

    ##
    ## plot
    ##
    fig, ax = plt.subplots(1,1)
    x = np.concatenate([x_vals_rising_edge,x_vals_falling_edge])
    y = np.concatenate([y_vals_rising_edge,y_vals_falling_edge])

    x_bw = 1
    y_bw = 10
    x_bins = np.arange(200,300 + x_bw, x_bw)
    y_bins = np.arange(0,600 + y_bw, y_bw)
    
    h = ax.hist2d(x, y,
        bins = (x_bins,y_bins),
        norm = matplotlib.colors.LogNorm(), cmap = plt.cm.YlOrRd)

    cb = fig.colorbar(h[-1])
    cb.set_label("Number of Pixels")

    fig.show()
    x = input()

    #fig, ax = plt.subplots(1,1)
    #markersize = 1.8
    #ax.plot(x_vals_rising_edge, y_vals_rising_edge, "ro",  markersize = markersize, label = "Rising")
    #ax.plot(x_vals_falling_edge, y_vals_falling_edge, "bo",markersize = markersize,  label = "Falling")
    #ax.set_xlim([200,300])
    #ax.legend(loc = "best")

    #fig.show()
    #x = input()

def plot_pixel_core_avg(data_map, tdac_map = None, tdac_select = None) :

    n_core_columns = 50
    n_core_rows = 48

    thresholds = sorted(data_map.keys())

    ##
    ## first make a map of core and pixel addresses
    ##

    core_averages = {}
    core_col_edges = []
    core_row_edges = []
    core_edges = []
    col_low = 0
    row_low = 0

    core_to_pixel_map = {}
    pixel_to_core_map = {}
    
    for i in range(n_core_columns) :
        edge = (col_low, col_low + 8)
        col_low += 8
        core_col_edges.append(edge)
    for i in range(n_core_rows) :
        edge = (row_low, row_low + 8)
        row_low += 8
        core_row_edges.append(edge)

    for icol, col_edges in enumerate(core_col_edges) :
        for irow, row_edges in enumerate(core_row_edges) :
            pix_columns = np.arange(col_edges[0], col_edges[1], 1)
            pix_rows = np.arange(row_edges[0], row_edges[1], 1)

            core_address = (icol, irow)
            if core_address not in core_to_pixel_map :
                core_to_pixel_map[core_address] = []

            for pix_column in pix_columns :
                for pix_row in pix_rows :
                    pix_address = (pix_column, pix_row)
                    core_to_pixel_map[core_address].append(pix_address)
                    pixel_to_core_map[pix_address] = core_address

    print(f"Loaded {len(core_to_pixel_map)} cores")
    print(f"Loaded {len(pixel_to_core_map)} pixels")

    x_vals_rising_edge, y_vals_rising_edge = np.array([]), np.array([])
    x_vals_falling_edge, y_vals_falling_edge = np.array([]), np.array([])

    ##
    ## now get the average toa and tot per core
    ##
    core_toa_avg_map = {} # map[threshold][core_address] = value
    core_tot_avg_map = {} # ditto
    core_toa_err_map = {} # ditto
    core_tot_err_map = {} # ditto

    toa_fwhm = []
    tot_fwhm = []

    n_pixels_plotted = 0

    do_tdac_selection = (tdac_map is not None) and (tdac_select != "")

    toa0 = []

    time_offset = None
    for ith, th in enumerate(thresholds) :
        tot_filename, toa_filename = data_map[th]

        core_toa_avg_map[th] = {}
        core_tot_avg_map[th] = {}
        core_toa_err_map[th] = {}
        core_tot_err_map[th] = {}

        n_pix_th = 0

        with open(toa_filename, "r") as toa_file, open(tot_filename, "r") as tot_file :

            all_toa_data = np.array(json.load(toa_file)["Data"])
            all_tot_data = np.array(json.load(tot_file)["Data"])

            core_0_address = None

            for icore, core_address in enumerate(core_to_pixel_map.keys()) :
                if icore == 0 :
                    core_0_address = core_address

                if icore > 500 : break
                pixel_addresses = core_to_pixel_map[core_address]
                if len(pixel_addresses) != 64 :
                    print(f"ERROR There were not 64 pixel addresses in core {icore} (core address: {core_address}")
                    sys.exit()
                min_pix_col, max_pix_col = min([x[0] for x in pixel_addresses]), max([x[0] for x in pixel_addresses])
                min_pix_row, max_pix_row = min([x[1] for x in pixel_addresses]), max([x[1] for x in pixel_addresses])
                if min_pix_col <= 1 :
                    min_pix_col = 2
                if max_pix_col >= 398 :
                    max_pix_col = 397

                core_toa_data = all_toa_data[min_pix_col:max_pix_col+1, min_pix_row:max_pix_row+1]
                core_tot_data = all_tot_data[min_pix_col:max_pix_col+1, min_pix_row:max_pix_row+1]
                tdac_sel = None
                if do_tdac_selection :
                    tdac_map_sel = tdac_map[min_pix_col:max_pix_col+1, min_pix_row:max_pix_row+1]

                # select pixels with the desired TDAC
                if do_tdac_selection :
                    idx_tdac = (tdac_map_sel == int(tdac_select))
                    core_toa_data = core_toa_data[idx_tdac]
                    core_tot_data = core_tot_data[idx_tdac]

                # nonzero (i.e. select pixels that had 100% occupancy during the FEScope scan)
                idx = core_toa_data > 0
                core_toa_data = core_toa_data[ idx ]
                core_tot_data = core_tot_data[ idx ]
                n_pix_th += core_toa_data.size

                # set it so that everything starts at 0
                if time_offset is None :
                    offset = np.mean(core_toa_data)
                    if not np.isnan(offset) :
                        time_offset = offset

                # compute the mean values and sigmas of offset corrected PToT and PToA data
                core_toa_avg = np.mean(core_toa_data)
                core_tot_avg = np.mean(core_tot_data)
                core_toa_err = np.std(core_toa_data)
                core_tot_err = np.std(core_tot_data)

                core_toa_avg_map[th][core_address] = core_toa_avg
                core_tot_avg_map[th][core_address] = core_tot_avg
                core_toa_err_map[th][core_address] = core_toa_err
                core_tot_err_map[th][core_address] = core_tot_err

                # set the core averages to all be equal to an arbitrarily selected core (here the first one)
                core_toa_data = core_toa_data + (core_toa_avg_map[th][core_0_address] - core_toa_data)

                ## subtract off the per-pixel average PToA when running a digital scan
                ## to remove effects of the digital logic
                ptoa_digital_scan = 190.5
                core_toa_data = core_toa_data - ptoa_digital_scan

                x_vals_rising_edge = np.concatenate([x_vals_rising_edge, core_toa_data])
                x_vals_falling_edge = np.concatenate([x_vals_falling_edge, core_toa_data + core_tot_data])
                y_vals_rising_edge = np.concatenate([y_vals_rising_edge, np.ones(len(core_toa_data)) * th])
                y_vals_falling_edge = np.concatenate([y_vals_falling_edge, np.ones(len(core_tot_data)) * th])

                # compute the FWHM, assuming that the pulse amplitude is 400 HARDCODED
                if th == 200 :
                    toa200 = np.mean(core_toa_data)
                    tot200 = np.mean(core_tot_data)
                    if not np.isnan(toa200) and not np.isnan(tot200) :
                        toa_fwhm.append(toa200)
                        tot_fwhm.append(tot200)

                #if th == min(thresholds) :
                if th >=50 and th <80 :
                    vals = x_vals_falling_edge[~np.isnan(x_vals_falling_edge)]
                    toa0.append( np.mean(vals) )
                    

        n_pixels_plotted = max([n_pix_th, n_pixels_plotted])

    print(f"N pixels plotted: {n_pixels_plotted}")

    toa0 = np.array(toa0)
    toa0 = toa0[~np.isnan(toa0)]
    x_max = int(1.1 * np.mean(toa0))

    #x_vals_rising_edge -= time_offset
    #x_vals_falling_edge -= time_offset

    ##
    ## FWHM information
    ##
    toa_fwhm_mean = np.mean(np.array(toa_fwhm))
    toa_fwhm_std = np.std(np.array(toa_fwhm))
    tot_fwhm_mean = np.mean(np.array(tot_fwhm))
    tot_fwhm_std = np.std(np.array(tot_fwhm))
    toa_fwhm = max(toa_fwhm) - min(toa_fwhm)
    tot_fwhm = max(tot_fwhm) - min(tot_fwhm)
    print(f"ToA Mean @ HM  : {toa_fwhm_mean * 1.5625} +/- {toa_fwhm_std * 1.5625} ns")
    print(f"ToT Mean @ HM  : {tot_fwhm_mean * 1.5625} +/- {tot_fwhm_std * 1.5625} ns")
    print(f"---")
    print(f"ToA FWHM       : {toa_fwhm * 1.5625} ns)")
    print(f"ToT FWHM       : {tot_fwhm * 1.5625} ns")


    ##
    ## plot
    ##
    fig, ax = plt.subplots(1,1)
    ax.set_facecolor("lightgrey")
    x = np.concatenate([x_vals_rising_edge, x_vals_falling_edge])
    y = np.concatenate([y_vals_rising_edge, y_vals_falling_edge])


    x_bw = 1
    y_bw = 10
    x_bins = np.arange(0, x_max + x_bw, x_bw)
    y_bins = np.arange(0, 500 + y_bw, y_bw)

    min_x = min(x_bins)
    max_x = max(x_bins)
    idx_for_height = (x > min_x) & (x < max_x)
    max_pulse_height = max(y[idx_for_height])
    print(f"Max pulse height = {max_pulse_height}")

    h, xedges, yedges, im = ax.hist2d(x * 1.5625, y, bins = (x_bins, y_bins), norm = matplotlib.colors.LogNorm(), cmap = plt.cm.YlOrRd)#, cmap = plt.cm.jet)#, norm = matplotlib.colors.LogNorm())#, cmap = plt.cm.YlOrRd)#, norm = matplotlib.colors.LogNorm(), cmap = plt.cm.YlOrRd)

    ax.set_xlabel("Time [ns]")
    ax.set_ylabel(r"$\Delta$th [counts]")

    cb = fig.colorbar(im)
    cb.set_label("Number of Pixels")

    fig.show()
    x = input()

def plot_diff_curves(input_directory) :

    parameter = ""
    subdir_naming = ""
    if "vff" in input_directory :
        parameter = "DiffVff"
        subdir_naming = "diffvff"
    elif "diffprecomp" in input_directory :
        parameter = "DiffPreComp"
        subdir_naming = "diffprecomp"
    else :
        print(f"ERROR Unknown parameter from input directory: {input_directory}")
        sys.exit(1)

    all_dirs = glob.glob(f"{input_directory}/{subdir_naming}_*")
    print(f"Found {len(all_dirs)} subdirs")
    parameter_values = []
    for dirname in all_dirs :
        param = dirname.strip().split("/")[-1].split("_")[-1]
        parameter_values.append(int(param))
    parameter_values = sorted(parameter_values)
    print(f"Parameter values: {parameter_values}")


    #parameter_values = [160]
    #parameter_values = [40, 80, 160, 320]
    parameter_values = [200, 500, 800]

    NUM_COLORS = len(parameter_values)
    fig, ax = plt.subplots(1,1)
    cm = plt.get_cmap('gist_rainbow')
    #ax.set_color_cycle([cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)])
    #cycle = []
    #for i in range(len(parameter_values)) :
    #    color = [cm(1.* i/NUM_COLORS)]
    #    cycle.append(color)
    #    cycle.append(color)
    #ax.set_prop_cycle(cycle)

    ax.set_xlabel("Time [ns]")
    ax.set_ylabel(r"$\Delta$th [counts]")
    ax.tick_params(which = "both", direction = "in", top = True, bottom = True, left = True, right = True)
        

#    h, xedges, yedges, im = ax.hist2d(x * 1.5625, y, bins = (x_bins, y_bins), norm = matplotlib.colors.LogNorm(), cmap = plt.cm.YlOrRd)#, cmap = plt.cm.jet)#, norm = matplotlib.colors.LogNorm())#, cmap = plt.cm.YlOrRd)#, norm = matplotlib.colors.LogNorm(), cmap = plt.cm.YlOrRd)

    avg_rising_edge_fix = {}
    for iparam, parameter_val in enumerate(parameter_values) :

        scan_dir = Path(f"{input_directory}/{subdir_naming}_{parameter_val}/last_scan")
        ok = scan_dir.exists() and scan_dir.is_dir()
        if not ok :
            print(f"ERROR Could not find scan dir: {scan_dir}")
            sys.exit(1)
        data_map = get_thresholds_from_input(str(scan_dir))
        thresholds = sorted(data_map.keys())

        x_vals_rising_edge, y_vals_rising_edge = [], []
        x_vals_falling_edge, y_vals_falling_edge = [], []

        n_nonzero_pixels = 0
        for ith, th in enumerate(thresholds) :
            tot_filename, toa_filename = data_map[th]
            with open(toa_filename, "r") as toa_file, open(tot_filename, "r") as tot_file :
                all_toa_data = np.array(json.load(toa_file)["Data"])
                all_tot_data = np.array(json.load(tot_file)["Data"])

                # remove LR columns
                all_toa_data = all_toa_data[2:398,:]
                all_tot_data = all_tot_data[2:398,:]

                # nonzero (i.e. select only pixels with 100%  occupancy)
                idx = all_toa_data > 0
                all_toa_data = all_toa_data[idx]
                all_tot_data = all_tot_data[idx]

                if iparam == 0 :
                    if th not in avg_rising_edge_fix :
                        avg_rising_edge_fix[th] = np.mean(all_toa_data)

                # set the core averages to all be equal to an arbitrarily selected core (here the first one)
                all_toa_data = all_toa_data + (avg_rising_edge_fix[th] - all_toa_data)

                # subtract off the per-pixel average PToT when running a digital scan
                # to remove effects of the digital logic
                ptoa_digital_scan = 190.5
                all_toa_data = all_toa_data - ptoa_digital_scan

                # we just want to plot points, so compute the mean
                x_rising_edge = np.mean(all_toa_data)
                x_falling_edge = x_rising_edge + np.mean(all_tot_data)

                x_vals_rising_edge.append(x_rising_edge)
                x_vals_falling_edge.append(x_falling_edge)
                y_vals_rising_edge.append(th)
                y_vals_falling_edge.append(th)

        # plot the data for this parameter
        hfig, hax = plt.subplots(1,1)

        x_bw = 1
        y_bw = 10
        xlo = 0
        xhi = 500
        ylo = 0
        yhi = 500
        x_bins = np.arange(xlo, xhi + x_bw, x_bw)
        y_bins = np.arange(ylo, yhi + y_bw, y_bw)

        ax.set_xlim([xlo,xhi])
        ax.set_ylim([ylo,yhi])

        #x = x_vals_rising_edge + x_vals_falling_edge
        #y = y_vals_rising_edge + y_vals_falling_edge


        h_rising, xedges, yedges, im = hax.hist2d( np.array(x_vals_rising_edge) * 1.5625, y_vals_rising_edge, bins = (x_bins, y_bins))
        h_falling, xedges, yedges, im = hax.hist2d( np.array(x_vals_falling_edge) * 1.5625, y_vals_falling_edge, bins = (x_bins, y_bins))


        # cut off at max pulse height
        x = np.concatenate([x_vals_rising_edge, x_vals_falling_edge])
        y = np.concatenate([y_vals_rising_edge, y_vals_falling_edge])
        idx_for_height = (x > xlo) & (x < xhi)
        max_pulse_height = max(y[idx_for_height])
        print(f"Max pulse height = {max_pulse_height}")


        for ih, h in enumerate([h_rising,  h_falling]) :
            x_vals, y_vals = [], []
            for x_bin_num in range(len(h)) :
                point_found = False
                x_val = 0
                y_val = 0
                y_vals_at_x = []
                found_x = False
                for ith, th in enumerate(thresholds) :
                    for y_bin_num in range(len(h[x_bin_num])) :
                        is_filled = (h[x_bin_num][y_bin_num] > 20)
                        if not is_filled : continue
                        y_val = yedges[y_bin_num]
                        if y_val != th : continue
                        #if len(y_vals) >=1 :
                        #    if y_val != (y_vals[-1] + 10) : continue
                        #if x_val in x_vals : continue
                        #x_vals.append(x_val)
                        y_vals_at_x.append(y_val)
                        #point_found = True
                        #break
                if len(y_vals_at_x) :
                    if len(y_vals_at_x) > 1 :
                        print(f"X = {xedges[x_bin_num]} : y_vals_at_x = {y_vals_at_x}")
                    
                    y_val = min(y_vals_at_x)

                    if y_val > 400 : continue
                    x_vals.append(xedges[x_bin_num])
                    y_vals.append(y_val)
                    #print(f"y_vals_at_x = {y_vals_at_x}")
                    #sys.exit()
                        
                        #print(f"pulse height = {y_val}, th = {th} -> {y_val == th}")
                        #continue
                        #if pulse_height != th : continue
                        #x_val = xedges[x_bin_num]
                        #y_val = th
                        #found_x = True
                        #break
                    #if point_found : break
                #for y_bin_num in range(len(h[x_bin_num])) :
                #    pulse_height = h[x_bin_num][y_bin_num]
                #    if pulse_height > 0 :
                #        x_val = xedges[x_bin_num]
                #        y_val = yedges[y_bin_num]
                #        if last_pulse_height >= 0 :
                #            if pulse_height == (last_pulse_height + 10) :
                #                x_vals.append(xedges[x_bin_num] + x_bw/2.0)
                #                y_vals.append(yedges[y_bin_num] + y_bw/2.0)
                #                if ih == 0 :
                #                    print(f"({xedges[x_bin_num]},{yedges[y_bin_num]}) -> {pulse_height}")
                #                last_pulse_height = pulse_height
            if ih == 0 :
                ax.plot(x_vals,y_vals,linestyle = "-", color = cm(1.0 * iparam/len(parameter_values)))
            else :
                ax.plot(x_vals,y_vals,linestyle = "-", label = f"{parameter}-{parameter_val}", color = cm(1.0 * iparam/len(parameter_values)))


#
#    print(f"len h = {len(h)}")
#    print(f"h[:1] = {h[:1]}")
#    print(f"len h[0] = {len(h[0])}")
#    x_vals, y_vals = [], []
#    for x_bin_num in range(len(h)) :
#        for y_bin_num in range(len(h[x_bin_num])) :
#            pulse_height = h[x_bin_num][y_bin_num]
#            if pulse_height > 0 :
#                x_vals.append(xedges[x_bin_num])
#                y_vals.append(yedges[y_bin_num])
#                print(f"bin[{x_bin_num},{y_bin_num}] = ({xedges[x_bin_num]},{yedges[y_bin_num]}) -> {pulse_height}")
#    fig, ax = plt.subplots(1,1)
#    ax.set_xlim([0, 300])
#    ax.set_ylim([0,500])
#    ax.plot(x_vals, y_vals)
#    fig.show()
#    _ = input()
#    sys.exit()
#    #print(f"ih = {ih}: mean x = {np.mean(x)}, mean y = {np.mean(y)}")
#    ax.set_xlabel("Time [ns]")
#    ax.set_ylabel(r"$\Delta$th [counts]")
#
#
#        x_bins = np.arange(xlo, xhi + x_bw, x_bw)
#        y_bins = np.arange(ylo, yhi + y_bw, y_bw)
#
#        h, xedges, yedges, im = ax.hist2d(np.array(x) * 1.5625, y, bins = (x_bins, y_bins))
#        print(f"h = {h}")
#        sys.exit()
#
#        ax.plot(x, y, linestyle = "-", label = f"{parameter}-{parameter_val}")

    ax.legend(loc = "best", frameon = False)
    fig.show()
    x = input()
    


def main() :

    parser = ArgumentParser()
    parser.add_argument("-i", "--input", default = "",
        required = True,
        help = "Provide an input file with JSON data files"
    )
    parser.add_argument("-p", "--pixel", default = "-1:-1",
        help = "Provide a single pixel address in the format <col>:<row>"
    )
    parser.add_argument("--tdac", default = "",
        help = "Provide a path to a TDAC map file"
    )
    parser.add_argument("--tdac-select", default = "",
        help = "Select pixels whose TDAC is the one provided"
    )
    args = parser.parse_args()

    p_input = Path(args.input)
    input_ok = p_input.exists() and p_input.is_dir()
    if not input_ok :
        print(f"ERROR: provided input directory (={args.input}) is bad")
        sys.exit(1)

    tdac_map = None
    if args.tdac != "" :
        tdac_map = load_tdac_map(args.tdac)

    data_map = get_thresholds_from_input(args.input)
    print(f"Found {len(data_map)} thresholds")

    selected_pix_col, selected_pix_row = [int(x) for x in args.pixel.split(":")]
    print(f"Plotting (col,row) = ({selected_pix_col},{selected_pix_row})")
    #plot_pixel(data_map, selected_pix_col, selected_pix_row)

    #plot_pixel_core_avg(data_map, tdac_map, args.tdac_select)
    plot_diff_curves(args.input)


if __name__ == "__main__" :
    main()
