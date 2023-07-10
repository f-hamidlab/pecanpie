# This class object loads, processes, visualises data output from suite2p
# Author                   : Jane Ling
# Date of creation         : 22/06/2023
# Date of last modification: 05/07/2023


# ------------------------------------------------------------------#
#                         load packages                             #
# ------------------------------------------------------------------#
import numpy as np
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import pandas as pd
import pickle

from skimage import measure
from skimage.measure import regionprops
from skimage.morphology import disk, binary_closing, binary_opening
import time

mpl.use('TkAgg')


# ------------------------------------------------------------------#
#                   class definition, load data                     #
# ------------------------------------------------------------------#
class s2p(object):

    def __init__(self, read_path, save_path=None, cells_to_process=None, cells_to_plot=None):
        """
        Start suite2p. Load .npy and .bin data. Define other parameters.

        Parameters
        ----------
        read_path : str
            Path to folder containing .npy and .bin files.

        save_path : str
            Path to folder for saving output files. (Default) same as read_path

        cells_to_process : list
            Indices to selected cells for data analysis. (Default) list of all ROIs identified in suite2p

        cells_to_plot : list
            Indices to selected cells for plotting. (Default) list of ROIs identified as cells in suite2p


        Returns
        -------
        None.

        """
        self.read_path = read_path
        print("Path: " + self.read_path)
        print("Reading .npy files...")

        # start timer
        t = Timer()

        # load .npy files if they exists
        self.F = self.read_npy('F.npy')
        self.Fneu = self.read_npy('Fneu.npy')
        self.spks = self.read_npy('spks.npy')
        self.stat = self.read_npy('stat.npy')
        self.ops = self.read_npy('ops.npy')
        try:
            self.ops = self.ops.item()
        except NameError:
            print("loading trimmed dataset...")

        self.iscell = self.read_npy('iscell.npy')
        # if none of the above exist, gives an error.
        if(all(x is None for x in [self.F, self.Fneu, self.spks, self.stat, self.ops, self.iscell])):
            raise Exception("No .npy file is found. Please check the data directory.")
        t.toc()  # print elapsed time

        # load .bin file if it exists
        if os.path.isfile(self.read_path + 'data.bin'):
            # opens the registered binary
            f = open(self.read_path + 'data.bin', 'rb')
            print("Reading .bin file...")
            self.bindata = np.fromfile(f, dtype=np.int16)
            f.close()
            t.toc()  # print elapsed time

            # reshaping bindata in the format of time, y, x
            print("Reshaping binary data...")
            self.bindata = np.reshape(self.bindata, (self.ops['nframes'], self.ops['Ly'], self.ops['Lx']))
            t.toc()  # print elapsed time

        else:
            self.bindata = None
            print("data.bin does not exist. Registered images are not loaded.")

        print("Defining other parameters...")
        # add save_path if it doesn't exist
        if save_path is not None:
            self.save_path = save_path
        else:
            self.save_path = read_path

        # make directory to contain figures and data outputs
        self.save_path_fig = self.save_path + "figures/"
        self.save_path_data = self.save_path + "data/"

        if not os.path.isdir(self.save_path):
            os.makedirs(self.save_path)
        if not os.path.isdir(self.save_path_fig):
            os.makedirs(self.save_path_fig)
        if not os.path.isdir(self.save_path_data):
            os.makedirs(self.save_path_data)

        # calculate delta F over F
        self.dfof = self.cal_dfof()

        # list of dictionaries to store image data for plot/saving
        self.im = [{}]

        # check cell selections and assign values
        self.cells_to_process = None
        self.cells_to_plot = None
        self.check_cells_to_process(cells_to_process)
        self.check_cells_to_plot(cells_to_plot)

        # DataFrame for storing metadata
        self.ori_metadata = pd.DataFrame()  # original metadata calculated by suite2p for all ROIs
        self.create_ori_metadata()  # initialize values from stat
        self.metadata = pd.DataFrame()  # initialize metadata of cells_to_process
        self.label_mask = []

        # TODO: add other fields if needed
        print('Done object initialization')
        t.toc()  # print elapsed time

    def read_npy(self, filename):
        """
        Loads data from .npy

        Parameters
        ----------
        filename : str
            filename of the .npy data file

        Returns
        -------
        data : ndarray (ROIs x timepoints)
            data stored in the .npy data file

        """
        if os.path.isfile(self.read_path + filename):
            data = np.load(self.read_path + filename, allow_pickle=True)
        else:
            data = None
            print("Warning: " + filename + " does not exist. Certain functions of this toolbox may be affected.")
        return data

    def check_cells_to_process(self, cells_to_process):
        # if cells_to_process is defined in input, use the defined value
        if cells_to_process is not None:
            # check if cells to be processed is within the limits of stat
            arr1 = set(range(len(self.stat)))
            arr2 = set(cells_to_process)
            if not arr1.union(arr2) == arr1:  # not a subset
                print("object.cells_to_process is not a subset of the recognised ROIs. Resetting to the default selection.")
                self.default_cells_to_process()
            else:
                self.cells_to_process = cells_to_process
        else:
            self.default_cells_to_process()

    def check_cells_to_plot(self, cells_to_plot):
        # if cells_to_plot is defined in input, use the defined value
        if cells_to_plot is not None:
            # check if cells to be plotted is a subset of cells to be processed
            arr1 = set(self.cells_to_process)
            arr2 = set(cells_to_plot)
            if not arr1.union(arr2) == arr1:  # not a subset
                print("object.cells_to_plot is not a subset of object.cells_to_process. Resetting to the default selection.")
                self.default_cells_to_plot()
            else:
                self.cells_to_plot = cells_to_plot
        else:
            self.default_cells_to_plot()

    def default_cells_to_process(self):
        if self.stat is not None:  # by default selected cells for processing are all ROIs
            self.cells_to_process = np.array(range(len(self.stat)))
        else:
            self.cells_to_process = None
            print("object.stat is not present. Please define object.cells_to_process for further processing and "
                  "plotting.")

    def default_cells_to_plot(self):
        if self.iscell is not None:  # by default selected cells for plotting are all real cells within cells_to_process
            k = np.nonzero(self.iscell[:, 0])
            k = np.array(k)
            k = k.reshape(-1)
            arr1 = set(self.cells_to_process)
            arr2 = set(k)
            self.cells_to_plot = arr1.intersection(arr2)
        else:
            self.cells_to_plot = None
            print("object.iscell is not present. Please define object.cells_to_plot for further processing and "
                  "plotting.")
    # ------------------------------------------------------------------#
    #                            pre-processing                         #
    # ------------------------------------------------------------------#
    def cal_dfof(self):
        """
        Calculates delta F over F.

        Parameters
        ----------
        None.

        Returns
        -------
        data : ndarray (ROIs x timepoints)
            delta F over F

        """
        # start timer
        t = Timer()

        # calculate delta F over F
        if (self.F is not None) & (self.Fneu is not None):
            print("Calculating delta F over F...")
            data = (self.F - self.Fneu) / self.Fneu
            t.toc()  # print elapsed time
        else:
            data = None
            print("F and/or Fneu does not exist. Cannot calculate delta F over F.")

        return data

        # ------------------------------------------------------------------#
        #                   create DataFrame for metadata                   #
        # ------------------------------------------------------------------#

    def create_ori_metadata(self):
        """
        Create a DataFrame for storing metadata of cells. Insert existing data from self.stat and self.iscell.

        Parameters
        ----------
        None.

        Returns
        -------
        None.

        """
        # start timer
        t = Timer()

        print('Creating dataframe from suite2p stat...')

        if (self.stat is not None) and (self.iscell is not None):
            def get_data_from_stat(ncells, item, col_name):
                data = np.zeros(ncells)
                for n in range(0, ncells):
                    data[n] = self.stat[n][item]
                self.ori_metadata[col_name] = data

            self.ori_metadata['iscell'] = self.iscell[:, 0]

            ncells = len(self.stat)  # include all the ROIs detected by suite2p
            get_data_from_stat(ncells, 'npix', 'area')
            get_data_from_stat(ncells, 'radius', 'radius')
            get_data_from_stat(ncells, 'aspect_ratio', 'aspect_ratio')
            get_data_from_stat(ncells, 'compact', 'compact')
            get_data_from_stat(ncells, 'solidity', 'solidity')

            t.toc()  # print elapsed time

        else:
            print("object.stat and/or object.iscell does not exist. Cannot create dataframe from suite2p stat.")

    def create_metadata(self):  # TODO: function to modify metadata when selection is changed
        """
        Calculate metadata of selected cells, columns include 'ROInum', 'iscell', 'ypix', 'xpix', 'contour', 'area',
        'centroid', 'major_axis', 'minor_axis', 'orientation', 'aspect_ratio', 'circularity', 'perimeter', 'compact',
         'solidity'

        Parameters
        ----------
        None.

        Returns
        -------
        None.

        """
        # start timer
        t = Timer()

        print('Calculating metadata...')
        # define columns for storing parameters to be calculated in later sessions
        d = {'ROInum': self.cells_to_process, 'ypix': None, 'xpix': None, 'contour': None, 'area': None,
             'perimeter': None, 'centroid': None, 'orientation': None, 'major_axis': None, 'minor_axis': None,
             'aspect_ratio': None, 'circularity': None, 'compact': None, 'solidity': None, 'iscell': None}
        self.metadata = pd.DataFrame(data=d)

        # this line allows the assignment of the array
        self.metadata = self.metadata.astype(object)
        label_mask = np.zeros((self.ops['Ly'], self.ops['Lx']))

        for n, m in enumerate(self.cells_to_process):
            # m is the index to the ROI in stat
            # n is the index to the ROI in the new metadata
            ypix = self.stat[m]['ypix'][~self.stat[m]['overlap']]
            xpix = self.stat[m]['xpix'][~self.stat[m]['overlap']]

            # for storing the image of individual cell, changes over loops
            im = np.zeros((self.ops['Ly'], self.ops['Lx']))
            im[ypix, xpix] = 1
            im = binary_closing(im, disk(4))  # size of disk set to 4 pixels for example data set. The size of disk
            # should be set according to the maximum 'hole' observed in the dataset.
            im = binary_opening(im, disk(2))  # size of disk set to 2 pixels, which is the width of axon in the
            # example dataset
            label_mask = label_mask + im * (n + 1)

        # get properties of the region and store in metadata
        regions = regionprops(label_mask.astype('int'))
        for props in regions:
            n = props.label-1
            self.metadata['centroid'][n] = props.centroid
            self.metadata['orientation'][n] = props.orientation
            self.metadata['major_axis'][n] = props.axis_major_length
            self.metadata['minor_axis'][n] = props.axis_minor_length
            self.metadata['perimeter'][n] = props.perimeter
            self.metadata['solidity'][n] = props.solidity
            self.metadata['area'][n] = props.num_pixels  # number of pixels
            self.metadata['iscell'][n] = self.iscell[self.cells_to_process[n]]  # whether the ROI is identified as a cell in suite2p
            self.metadata['contour'][n] = np.squeeze(np.array(measure.find_contours(label_mask == n+1)))  # contour of ROIs

        # Trimming metadata to remove zero entries
        # area of cell has to be more than 1 pixel for calculation of major and minor axis
        area_threshold = 1

        def switch_val(x):
            return new_vals[old_vals.index(x)] if x in old_vals else x
        old_vals = list(self.metadata[self.metadata['area'] <= area_threshold]['ROInum']+1)
        new_vals = list(np.zeros(len(old_vals)))
        vc = np.vectorize(switch_val)
        self.label_mask = vc(label_mask)

        self.metadata = self.metadata.loc[self.metadata['area'] > area_threshold]
        self.cells_to_process = np.array(self.metadata['ROInum']).astype('int')


        # calculations for other items in metadata
        self.metadata['aspect_ratio'] = self.metadata['major_axis']/self.metadata['minor_axis']
        # TODO: fix circularity and compact
        # self.metadata['circularity'] = 4*np.pi * np.dot(self.metadata['area']/(self.metadata['perimeter'] ^ 2))

        t.toc()  # print elapsed time

    def change_cell_selection(self, cells_to_process=None, cells_to_plot=None):
        self.check_cells_to_process(cells_to_process)
        self.check_cells_to_plot(cells_to_plot)
        self.create_metadata()


    # ------------------------------------------------------------------#
    #                           data analysis                           #
    # ------------------------------------------------------------------#

    # ------------------------------------------------------------------#
    #                          data visualisation                       #
    # ------------------------------------------------------------------#
    def im_plot(self, plottype, plot=True, filename=None):  # TODO: add other parameters for more flexible plot
        """
        Sets parameters for plotting according to plot type.

        Parameters
        ----------
        plottype : str
            'avg_bin' = plots the registered binary data averaged over time
            'selected_cells' = plots the selected cells in peak delta F over F intensity
            'contour' = plots the selected cells with their contours after morphological operations
            'axis' = plots the selected cells with their contours and axes after morphological operations
        plot : bool
            Whether to show plot or not. (Default) True
        filename : str
            Filename of figure to save. If filename is not set, the figure will NOT be saved. (Default) None

        Returns
        -------
        None.

        """
        print("Initializing plot for " + plottype + "...")
        t = Timer()

        # check the current number of plots
        if not self.im[0]:  # the first dictionary is empty
            k = 0
        else: # the first dictionary is NOT empty. Append to the list of dictionaries
            k = len(self.im)
            self.im.append({})

        # for plotting of the average binary image
        if plottype == 'avg_bin':
            if self.bindata is not None:
                self.im[k]['imdata'] = np.mean(self.bindata, axis=0)
                self.im[k]['title'] = "Motion-Corrected Image"
                self.im[k]['cmap'] = 'gray'
                self.im[k]['type'] = 'image'

            else:
                print("data.bin does not exist. Plot type is not supported.")

        # for plotting ROIs in peak delta F over F intensity, the mask for ROI is from output of suite2p
        elif plottype == 'selected_cells':
            self.im[k]['imdata'] = self.switch_idx_to_intensity()
            self.im[k]['title'] = "Selected cells at peak intensity"
            self.im[k]['cmap'] = 'gray'
            self.im[k]['type'] = 'image'

        # for plotting ROIs with their contours after morphological operations
        elif plottype == 'contour':
            self.im[k]['line_data'] = []
            for n in self.cells_to_plot:
                idx = self.metadata.index[self.metadata['ROInum'] == n]  # the index to the ROI in metadata

                # get the contour of ROI
                contour = self.metadata.loc[idx]['contour'].values[0]
                self.im[k]['line_data'].append({'x': contour[:, 1], 'y': contour[:, 0]})

            self.im[k]['imdata'] = self.switch_idx_to_intensity()
            self.im[k]['title'] = "Contours of selected cells"
            self.im[k]['cmap'] = 'gray'
            self.im[k]['type'] = 'image & line'

        # for plotting ROIs with their contours and axes after morphological operations
        elif plottype == 'axis':
            self.im[k]['line_data'] = []

            for n in self.cells_to_plot:
                idx = self.metadata.index[self.metadata['ROInum'] == n]  # the index to the ROI in metadata

                # get the contour of ROI
                contour = self.metadata.loc[idx]['contour'].values[0]
                self.im[k]['line_data'].append({'x': contour[:, 1], 'y': contour[:, 0]})

                # for plotting the major and minor axes
                centroid = self.metadata.loc[idx]['centroid'].values[0]
                y0, x0 = centroid
                orientation = self.metadata.loc[idx]['orientation'].values[0]
                major_axis = self.metadata.loc[idx]['major_axis'].values[0]
                minor_axis = self.metadata.loc[idx]['minor_axis'].values[0]

                x1 = x0 + math.cos(orientation) * 0.5 * minor_axis
                y1 = y0 - math.sin(orientation) * 0.5 * minor_axis
                x2 = x0 - math.sin(orientation) * 0.5 * major_axis
                y2 = y0 - math.cos(orientation) * 0.5 * major_axis

                self.im[k]['line_data'].append({'x': (x0, x1), 'y': (y0, y1), 'linestyle': '-r', 'linewidth': 1})
                self.im[k]['line_data'].append({'x': (x0, x2), 'y': (y0, y2), 'linestyle': '-r', 'linewidth': 1})
            self.im[k]['imdata'] = self.switch_idx_to_intensity()
            self.im[k]['title'] = "Axis of fitted ellipse"
            self.im[k]['cmap'] = 'gray'
            self.im[k]['type'] = 'image & line'

        else:
            print("Plot type is undefined.")
            return 0

        # determining the canvas
        if k == 0:  # the first plot
            self.im[k]['canvas'] = 0
        elif self.im[k - 1]['plot'] == 0:  # the previous plot is not shown
            self.im[k]['canvas'] = self.im[k - 1]['canvas']
        else:
            self.im[k]['canvas'] = self.im[k - 1]['canvas'] + 1

        self.im[k]['xlabel'] = '(pixels)'
        self.im[k]['ylabel'] = '(pixels)'
        self.im[k]['xlim'] = [0, self.im[k]['imdata'].shape[1] - 1]
        self.im[k]['ylim'] = [0, self.im[k]['imdata'].shape[0] - 1]

        self.im[k]['plot'] = plot
        self.im[k]['filename'] = filename

        t.toc()

    def switch_idx_to_intensity(self):
        def switch_val(x):
            return new_vals[old_vals.index(x)] if x in old_vals else x

        # get max intensity for each cell
        max_dfof = np.max(self.dfof, axis=1)  # axis = 1 for the time dimension

        # substitute cells in process but not in plot with 0
        cells_not_to_plot = np.array(list(filter(lambda x: x not in self.cells_to_plot, self.cells_to_process))).astype('int')
        max_dfof[cells_not_to_plot] = 0

        # substitute cells to process with intensity
        max_dfof = max_dfof[self.cells_to_process]
        old_vals = list(self.cells_to_process+1)
        new_vals = list(max_dfof)
        vc = np.vectorize(switch_val)

        return vc(self.label_mask)

    def plot_fig(self):
        """
        Visualize image data and saving

        Parameters
        ----------
        None.

        Returns
        -------
        None.

        """
        for k in range(0, len(self.im)):
            if self.im[k]['plot'] is True:
                plt.figure(self.im[k]['canvas'])
                plt.title(self.im[k]['title'])
                plt.xlabel(self.im[k]['xlabel'])
                plt.ylabel(self.im[k]['ylabel'])
                plt.xlim(self.im[k]['xlim'])
                plt.ylim(self.im[k]['ylim'])

                if self.im[k]['type'] == 'image':
                    plt.imshow(self.im[k]['imdata'], cmap=self.im[k]['cmap'])

                elif self.im[k]['type'] == 'image & line':
                    plt.imshow(self.im[k]['imdata'], cmap=self.im[k]['cmap'])

                    num_lines = len(self.im[k]['line_data'])
                    for n in range(num_lines):
                        x = self.im[k]['line_data'][n]['x']
                        y = self.im[k]['line_data'][n]['y']

                        if 'linestyle' in self.im[k]['line_data'][n]:
                            linestyle = self.im[k]['line_data'][n]['linestyle']
                        else:
                            linestyle = None

                        if 'linewidth' in self.im[k]['line_data'][n]:
                            linewidth = self.im[k]['line_data'][n]['linewidth']
                        else:
                            linewidth = None

                        if linestyle is None:
                            plt.plot(x, y, linewidth=linewidth)
                        else:
                            plt.plot(x, y, linestyle, linewidth=linewidth)

            if self.im[k]['filename'] is not None:  # save as a tif file if 'filename' has been assigned
                if not self.im[k]['filename'].endswith(".tif"):
                    self.im[k]['filename'] = self.im[k]['filename'] + ".tif"
                plt.savefig(self.save_path_fig + self.im[k]['filename'])

        plt.show()


# ------------------------------------------------------------------#
#                               Timer                               #
# ------------------------------------------------------------------#


class Timer:
    """
    Timer for checking performance

    Parameters
    ----------
    None.

    Returns
    -------
    None.

    """

    def __init__(self):
        self._start_time = time.perf_counter()

    def tic(self):
        # Start a new timer
        self._start_time = time.perf_counter()

    def toc(self):
        # Stop the timer, and report the elapsed time
        elapsed_time = time.perf_counter() - self._start_time
        print(f"Done. Elapsed time: {elapsed_time:0.4f} seconds")
        self._start_time = time.perf_counter()
