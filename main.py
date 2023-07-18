# This script contains an example pipeline for processing suite2p output
# Author: Jane Ling
# Date of creation: 20/06/2023
# Date of last modification: 14/07/2023
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
s2p = pp.PecanPie(read_path, verbal=False)  # for test dataset
s2p.create_metadata()

# ------------------------------------------------------------------#
#            change selection of ROI from figure (optional)         #
# ------------------------------------------------------------------#

# s2p.cells_to_process_from_fig()
# s2p.cells_to_plot_from_fig()
