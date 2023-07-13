# This script contains an example pipeline for processing suite2p output
# Author: Jane Ling
# Date of creation: 20/06/2023
# Date of last modification: 11/07/2023
# Scope: For single-channel, single-plane images only

# ------------------------------------------------------------------#
#                         load packages                             #
# ------------------------------------------------------------------#
import suite2p_tools as st
import numpy as np

# ------------------------------------------------------------------#
#                            load data                              #
# ------------------------------------------------------------------#
# set filepath to test dataset
filepath = 'testdata/originals/'
savepath = 'testdata/outputs/'

# ------------------------------------------------------------------#
#                     predefine ROIs (optional)                     #
# ------------------------------------------------------------------#
# Selecting cells to be processed. If set to None, process all ROIs
# cells_to_process = np.linspace(0, 49, endpoint=False, dtype='int')
cells_to_process = [0, 1, 2, 3, 4, 5]
# Selecting cells to be plotted
# cells_to_plot = [2, 3]

# ------------------------------------------------------------------#
#                          creating object                          #
# ------------------------------------------------------------------#
# load .npy bindata from read_path
# s2p = st.s2p(filepath, savepath)  # for test dataset
# s2p = st.s2p(filepath, savepath, cells_to_process, cells_to_plot)  # for test dataset
s2p = st.s2p(filepath, savepath, cells_to_process)  # for test dataset
# s2p = st.s2p(filepath, savepath, cells_to_plot)  # for test dataset
s2p.create_metadata()

# ------------------------------------------------------------------#
#       change selection of ROI by list of numbers (optional)       #
# ------------------------------------------------------------------#
# # Selecting cells to be processed. If set to None, process all ROIs
# # cells_to_process = np.linspace(0, 5, 4, endpoint=False, dtype='int')
# cells_to_process = [0, 1, 2, 3, 4, 5]
#
# # Selecting cells to be plotted
# cells_to_plot = [0, 2]
# s2p.change_cell_selection(cells_to_process, cells_to_plot)
# # Or for just change in one set of selection, use:
# # s2p.change_cell_selection(cells_to_process)
# # s2p.change_cell_selection(cells_to_plot)


# ------------------------------------------------------------------#
#                           plotting figures                        #
# ------------------------------------------------------------------#
# # initialize plot average fluorescence
# s2p.im_plot('avg_bin', plot=True, filename='average_binary.tif')
#
# # initialize plot for selected cells at peak intensity
# s2p.im_plot('selected_cells', plot=True, filename='selected_cells.tif')
#
# # initialize plot for selected cells with contours
# s2p.im_plot('contour', plot=True, filename='contours.tif')
#
# # initialize plot for selected cells with contours and major and minor axes
# s2p.im_plot('axis', plot=True, filename='axes.tif')
#
# # plot and save figures
# s2p.plot_fig()

# ------------------------------------------------------------------#
#            change selection of ROI from figure (optional)         #
# ------------------------------------------------------------------#

s2p.cells_to_process_from_fig()
# s2p.cells_to_plot_from_fig()

# # initialize plot for selected cells at peak intensity
# s2p.im_plot('selected_cells', plot=True, filename='selected_cells.tif')
#
# # plot and save figures
# s2p.plot_fig()





