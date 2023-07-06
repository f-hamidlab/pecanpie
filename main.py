# This script contains an example pipeline for processing suite2p output
# Author: Jane Ling
# Date of creation: 20/06/2023
# Date of last modification: 05/07/2023
# Scope: For single-channel, single-plane images only

# ------------------------------------------------------------------#
#                         load packages                             #
# ------------------------------------------------------------------#
import suite2p_tools as st

# ------------------------------------------------------------------#
#                            load data                              #
# ------------------------------------------------------------------#
# set filepath to outputs folder of original dataset
# filepath = '/media/cdn-bc/RAID/Datasets/Harris_BioRxiv_2016_suite2p/outputs/suite2p/plane0/'

# set filepath to test dataset
filepath = '/home/jane/PycharmProjects/Suite2p/testdata/originals/'
savepath = '/home/jane/PycharmProjects/Suite2p/testdata/outputs/'

# load .npy bindata from read_path
# s2p = st.s2p(filepath)  # for original dataset
s2p = st.s2p(filepath, savepath)  # for test dataset

# s2p.cells_to_plot = [7, 8]  # for selected specific cells to be plotted

# initialize plot average fluorescence
s2p.im_plot('avg_bin', plot=True, filename='average_binary.tif')

# initialize plot for selected cells at peak intensity
s2p.im_plot('selected_cells', plot=True, filename='selected_cells.tif')

# initialize plot for selected cells with contours
s2p.im_plot('contour', plot=True, filename='contours.tif')

# initialize plot for selected cells with contours and major and minor axes
s2p.im_plot('axis', plot=True, filename='axes.tif')

# plot and save figures
s2p.plot_fig()






