# This class object loads, processes, visualises data output from suite2p
# Author                   : Jane Ling
# Date of creation         : 22/06/2023
# Date of last modification: 28/06/2023

# ------------------------------------------------------------------#
#                         load packages                             #
# ------------------------------------------------------------------#
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import pandas as pd
import pickle
import time

mpl.use('TkAgg')


# ------------------------------------------------------------------#
#                   class definition, load data                     #
# ------------------------------------------------------------------#
class s2p(object):

    def __init__(self, read_path, save_path=None):
        """
        Start suite2p. Load .npy and .bin data. Define other parameters.

        Parameters
        ----------
        read_path : str
            Path to folder containing .npy and .bin files.

        save_path : str
            Path to folder for saving output files. (Default) same as read_path

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
        if (self.F is None) & (self.Fneu is None) & (self.spks is None) & (self.stat is None) & (self.ops is None) \
                & (self.iscell is None):
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

            # trimming image to usable field-of-view
            # TODO: check that the trimmed data match the coordinates of cells in stats
            # print("Trimming binary data...")
            # self.bindata = self.bindata[:, self.ops['yrange'][0]: self.ops['yrange'][1],
            #                self.ops['xrange'][0]:self.ops['xrange'][1]]
            # t.toc()  # print elapsed time
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

        # list of dictionaries to store image data for plot/saving
        self.im = [{}]

        # DataFrame for storing metadata of ROIs
        self.metadata = pd.DataFrame()
        self.create_dataframe()

        # by default selected cells are all real cells
        if self.iscell is not None:
            k = np.nonzero(self.iscell[:, 0])
            k = np.array(k)
            self.selected_cells = k.reshape(-1)
        else:
            self.selected_cells = None
            print("object.iscell is not present. Please define object.selected_cells for further processing and "
                  "plotting.")

        # TODO: add other fields if needed
        t.toc()  # print elapsed time

        # calculate delta F over F
        self.dfof = self.cal_dfof()

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
    #                           data analysis                           #
    # ------------------------------------------------------------------#

    # ------------------------------------------------------------------#
    #                   create DataFrame for metadata                   #
    # ------------------------------------------------------------------#

    def create_dataframe(self):
        """
        Create a DataFrame for storing metadata of cells. Insert existing data from self.stat and self.iscell.

        Parameters
        ----------
        None.

        Returns
        -------
        None.

        """
        def get_data_from_stat(ncells, item, col_name):
            data = np.zeros(ncells)
            for n in range(0, ncells):
                data[n] = self.stat[n][item]
            self.metadata[col_name] = data

        self.metadata['iscell'] = self.iscell[:, 0]

        ncells = len(self.stat)
        get_data_from_stat(ncells, 'npix', 'area')
        get_data_from_stat(ncells, 'radius', 'radius')
        get_data_from_stat(ncells, 'aspect_ratio', 'aspect_ratio')
        get_data_from_stat(ncells, 'compact', 'compact')
        get_data_from_stat(ncells, 'solidity', 'solidity')


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
        else:
            k = len(self.im)
            self.im.append({})

        if plottype == 'avg_bin':
            if self.bindata is not None:
                self.im[k]['data'] = np.mean(self.bindata, axis=0)
                self.im[k]['title'] = "Motion-Corrected Image"
                self.im[k]['cmap'] = 'gray'
                self.im[k]['type'] = 'image'

            else:
                print("data.bin does not exist. Plot type is not supported.")

        elif plottype == 'selected_cells':
            im = np.zeros((self.ops['Ly'], self.ops['Lx']))
            ncells = len(self.selected_cells)
            for n in range(0, ncells):
                m = self.selected_cells[n]
                ypix = self.stat[m]['ypix'][~self.stat[m]['overlap']]
                xpix = self.stat[m]['xpix'][~self.stat[m]['overlap']]
                im[ypix, xpix] = self.iscell[m, 0] * max(self.dfof[m, :])

            self.im[k]['data'] = im
            self.im[k]['title'] = "Selected cells at peak intensity"
            self.im[k]['cmap'] = 'gray'
            self.im[k]['type'] = 'image'

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
        self.im[k]['xlim'] = [0, self.im[k]['data'].shape[1] - 1]
        self.im[k]['ylim'] = [0, self.im[k]['data'].shape[0] - 1]

        self.im[k]['plot'] = plot
        self.im[k]['filename'] = filename

        t.toc()

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
                plt.imshow(self.im[k]['data'], cmap=self.im[k]['cmap'])
                plt.title(self.im[k]['title'])
                plt.xlabel(self.im[k]['xlabel'])
                plt.ylabel(self.im[k]['ylabel'])
                plt.xlim(self.im[k]['xlim'])
                plt.ylim(self.im[k]['ylim'])

            if self.im[k]['filename'] is not None:
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
        """Start a new timer"""
        self._start_time = time.perf_counter()

    def toc(self):
        """Stop the timer, and report the elapsed time"""
        elapsed_time = time.perf_counter() - self._start_time
        print(f"Done. Elapsed time: {elapsed_time:0.2f} seconds")
        self._start_time = time.perf_counter()
