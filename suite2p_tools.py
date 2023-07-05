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
from skimage.measure import regionprops
from skimage.morphology import disk, binary_closing, binary_opening
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

        # calculate delta F over F
        self.dfof = self.cal_dfof()

        # list of dictionaries to store image data for plot/saving
        self.im = [{}]

        # by default selected cells for plotting are all real cells
        if self.iscell is not None:
            k = np.nonzero(self.iscell[:, 0])
            k = np.array(k)
            self.cells_to_plot = k.reshape(-1)
        else:
            self.cells_to_plot = None
            print("object.iscell is not present. Please define object.cells_to_plot for further processing and "
                  "plotting.")

        # by default selected cells for processing are all ROIs
        if self.stat is not None:
            self.cells_to_process = np.array(range(len(self.stat)))
        else:
            self.cells_to_process = None
            print("object.stat is not present. Please define object.cells_to_process for further processing and "
                  "plotting.")

        # DataFrame for storing metadata of ROIs
        self.ori_metadata = pd.DataFrame()  # original metadata calculated by suite2p for all ROIs
        self.create_ori_metadata()  # initialize values from stat

        self.metadata = pd.DataFrame()  # metadata of cells_to_process
        self.create_metadata()  # initialize values with calculations on stat

        # TODO: add other fields if needed

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

            ncells = len(self.stat)
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

        ncells = len(self.cells_to_process)
        im0 = np.zeros((self.ops['Ly'], self.ops['Lx']))  # for storing the overall image
        im2 = im0

        for n in range(ncells):
            m = self.cells_to_process[n]
            ypix = self.stat[m]['ypix'][~self.stat[m]['overlap']]
            xpix = self.stat[m]['xpix'][~self.stat[m]['overlap']]
            # for storing the image of individual cell, changes over loops
            im1 = np.zeros((self.ops['Ly'], self.ops['Lx']))
            im1[ypix, xpix] = 1
            im1 = binary_closing(im1, disk(4))  # size of disk set to 4 pixels for example data set. The size of disk
            # should be set according to the maximum 'hole' observed in the dataset.
            im1 = binary_opening(im1, disk(2))  # size of disk set to 2 pixels, which is the width of axon in the
            # example dataset

            new_mask = np.transpose(np.array(np.nonzero(im1)))
            self.metadata['ypix'][n] = new_mask[:, 0]
            self.metadata['xpix'][n] = new_mask[:, 1]
            self.metadata['area'][n] = len(new_mask)
            self.metadata['iscell'][n] = self.iscell[m]

            regions = regionprops(im1.astype('int'))
            for props in regions:
                self.metadata['centroid'][n] = props.centroid
                self.metadata['orientation'][n] = props.orientation
                self.metadata['major_axis'][n] = props.axis_major_length
                self.metadata['minor_axis'][n] = props.axis_minor_length
                self.metadata['perimeter'][n] = props.perimeter
                self.metadata['solidity'][n] = props.solidity

            if n in self.cells_to_plot:
                im0 = im0 + im1 * max(self.dfof[n, :])
                im2 = im2 + im1 * (n+1)  # +1 such that the first ROI won't be labeled as zero

        self.metadata['aspect_ratio'] = self.metadata['major_axis']/self.metadata['minor_axis']
        # TODO: fix circularity and compact
        # self.metadata['circularity'] = 4*np.pi * np.dot(self.metadata['area']/(self.metadata['perimeter'] ^ 2))

        t.toc()  # print elapsed time
        # TODO: move to plotting
        # plt.figure(3)
        # plt.imshow(im0, cmap=plt.cm.gray)
        # plt.title('Filled selected cells')
        # plt.xlabel('(pixels)')
        # plt.ylabel('(pixels)')
        # ax = plt.gca()
        # ax.invert_yaxis()
        #
        # print('Displaying masks of cells...')
        # # Display the image and plot all contours found
        # plt.figure(2)
        # plt.imshow(im0, cmap=plt.cm.gray)
        # plt.title('Contours of selected cells')
        # plt.xlabel('(pixels)')
        # plt.ylabel('(pixels)')
        # ax = plt.gca()
        # ax.invert_yaxis()
        # for n in range(ncells):
        #     m = self.cells_to_process[n]
        #     if m in self.cells_to_plot:
        #         plt.plot(self.metadata['contour'][m][:, 1], self.metadata['contour'][m][:, 0], linewidth=1)
        #
        # plt.show()
        # im2 = im2.astype('int')
        # regions = regionprops(im2)
        # plt.figure(4)
        # plt.imshow(im0, cmap=plt.cm.gray)
        # plt.title('Axis of fitted ellipse')
        # plt.xlabel('(pixels)')
        # plt.ylabel('(pixels)')
        #
        # for props in regions:
        #     y0, x0 = props.centroid
        #     orientation = props.orientation
        #     x1 = x0 + math.cos(orientation) * 0.5 * props.axis_minor_length
        #     y1 = y0 - math.sin(orientation) * 0.5 * props.axis_minor_length
        #     x2 = x0 - math.sin(orientation) * 0.5 * props.axis_major_length
        #     y2 = y0 - math.cos(orientation) * 0.5 * props.axis_major_length
        #
        #     plt.plot((x0, x1), (y0, y1), '-r', linewidth=1)
        #     plt.plot((x0, x2), (y0, y2), '-r', linewidth=1)
        #     plt.plot(x0, y0, '.g', markersize=3)
        #
        #     minr, minc, maxr, maxc = props.bbox
        #     bx = (minc, maxc, maxc, minc, minc)
        #     by = (minr, minr, maxr, maxr, minr)
        #     plt.plot(bx, by, '-b', linewidth=1)
        # ax = plt.gca()
        # ax.invert_yaxis()
        # plt.show()

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
            ncells = len(self.cells_to_plot)
            for n in range(0, ncells):
                m = self.cells_to_plot[n]
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
        print(f"Done. Elapsed time: {elapsed_time:0.4f} seconds")
        self._start_time = time.perf_counter()
