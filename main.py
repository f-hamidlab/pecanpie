# This script contains an example pipeline for processing suite2p output
# Author: Jane Ling
# Date of creation: 20/06/2023
# Date of last modification: 16/08/2023
# Scope: For single-channel, single-plane images only

# ------------------------------------------------------------------#
#                         load packages                             #
# ------------------------------------------------------------------#
import pecan_pie as pp

# ------------------------------------------------------------------#
#                            load data                              #
# ------------------------------------------------------------------#
# set read_path to test dataset
read_path = 'testdata/originals/'

# ------------------------------------------------------------------#
#                          creating object                          #
# ------------------------------------------------------------------#
# load .npy bindata from read_path and initiate pecanpie object
s2p = pp.PecanPie(read_path, verbose=True)
s2p.create_metadata()

# ------------------------------------------------------------------#
#                    plotting figures (optional)                    #
# ------------------------------------------------------------------#
# initialize plot for selected cells at peak intensity
# s2p.create_fig('user_defined', plot=True, data='area')

# plot and save figures
# s2p.plot_fig()

# ------------------------------------------------------------------#
#            change selection of ROI from figure (optional)         #
# ------------------------------------------------------------------#
# s2p.cells_to_process_from_fig()
# s2p.cells_to_plot_from_fig()

# ------------------------------------------------------------------#
#               principle component analysis (optional)             #
# ------------------------------------------------------------------#
s2p.pca()
