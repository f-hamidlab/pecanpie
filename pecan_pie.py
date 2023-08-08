# This class object loads, processes, visualises data output from suite2p
# Author                   : Jane Ling
# Date of creation         : 22/06/2023
# Date of last modification: 19/07/2023

# ------------------------------------------------------------------#
#                         load packages                             #
# ------------------------------------------------------------------#

import numpy as np
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import mplcursors
import os
import pandas as pd

# import pickle
from rich.console import Console
from rich.table import Table
from skimage import measure
from skimage.measure import regionprops
from skimage.morphology import disk, binary_closing, binary_opening
from skimage.segmentation import find_boundaries
import time

mpl.use('TkAgg')  # need this line, otherwise a pycharm console error would occur


# ------------------------------------------------------------------#
#                   class definition, load data                     #
# ------------------------------------------------------------------#

class PecanPie(object):
    def __init__(self, read_path, save_path=None, cells_to_process=None, cells_to_plot=None, verbose=False):
        """
        Start PecanPie. Load .npy and .bin data. Define other parameters.

        Parameters
        ----------
        read_path : str
            Path to folder containing .npy and .bin files.

        save_path : str
            Path to folder for saving output files. (Default) same as read_path

        cells_to_process : array
            Indices to selected cells for data analysis. (Default) All ROIs identified in suite2p

        cells_to_plot : array
            Indices to selected cells for plotting. (Default) All ROIs within cells_to_process that are identified as
            cells in suite2p


        Returns
        -------
        None.

        """
        self._verbose = verbose
        self.read_path = read_path
        print(_BColours.HEADER, "Path: " + self.read_path, _BColours.ENDC)

        # start timer
        t = _Timer(self._verbose)
        t.tic("Reading .npy files...")

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
        if all(x is None for x in [self.F, self.Fneu, self.spks, self.stat, self.ops, self.iscell]):
            raise Exception("No .npy file is found. Please check the data directory.")
        t.toc()  # print elapsed time

        # load .bin file if it exists
        if os.path.isfile(self.read_path + 'data.bin'):
            # opens the registered binary
            f = open(self.read_path + 'data.bin', 'rb')
            t.tic("Reading .bin file...")
            self.bindata = np.fromfile(f, dtype=np.int16)
            f.close()
            t.toc()  # print elapsed time

            # reshaping bindata in the format of time, y, x
            t.tic("Reshaping binary data...")
            self.bindata = np.reshape(self.bindata, (self.ops['nframes'], self.ops['Ly'], self.ops['Lx']))
            t.toc()  # print elapsed time

        else:
            self.bindata = None
            print(_BColours.WARNING, "data.bin does not exist. Registered images are not loaded.", _BColours.ENDC)

        # calculate delta F over F
        self.dfof = self.cal_dfof()

        t.tic("Defining other parameters...")

        # add save_path if it doesn't exist
        if save_path is not None:
            self.save_path = save_path
        else:
            k = os.path.split(os.path.split(read_path)[0])
            self.save_path = k[0] + "/outputs/"

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
        self._im = [{}]

        # check cell selections and assign values
        self.cells_to_process = None
        self.cells_to_plot = None
        self.check_cells_to_process(cells_to_process)
        self.check_cells_to_plot(cells_to_plot)

        # _color scheme / plot properties
        self._color = {'red': 'salmon', 'green': 'seagreen'}
        self._linewidth = 2
        create_cmap("#2E8B57")  # colormap for contours, #2E8B57 is seagreen
        self.overlay_handle = None

        # temporary data slot for passing internal variables
        self._tmp = []

        # TODO: add other fields if needed
        self.table = {}
        t.toc()  # print elapsed time

        # DataFrame for storing metadata
        self.ori_metadata = pd.DataFrame()  # original metadata calculated by suite2p for all ROIs
        self.create_ori_metadata()  # initialize values from stat
        self.metadata = pd.DataFrame()  # initialize metadata of cells_to_process
        self.label_mask = []
        self.contour_mask = []

        # Display some data about the object
        if self._verbose:
            print(_BColours.OKGREEN, 'Done object initialization.', _BColours.ENDC)

    def __repr__(self):
        """
        Print information about the PecanPie object when the name of object is typed in the console.

        Parameters
        ----------
        None.

        Returns
        -------
        None.

        """
        print(_BColours.HEADER, "Path: " + self.read_path, _BColours.ENDC)
        print_data_status(self.F is not None, 'self.F')
        print_data_status(self.Fneu is not None, 'self.Fneu')
        print_data_status(self.spks is not None, 'self.spks')
        print_data_status(self.stat is not None, 'self.stat')
        print_data_status(self.ops is not None, 'self.ops')
        print_data_status(self.bindata is not None, 'self.bindata')

        print("\n")
        print_data_status(self.ops['Lx'], 'Nx')
        print_data_status(self.ops['Ly'], 'Ny')
        print_data_status(self.F.shape[1], 'Timepoints')

        print("\n")
        self.print_metadata()

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
            print(_BColours.WARNING, "Warning: " + filename + "does not exist. Certain functions of this toolbox may "
                                                              "be affected.", _BColours.ENDC)
        return data

    def check_cells_to_process(self, cells_to_process):
        """
        Check that the newly defined cells_to_process is within the scope of stat.npy

        Parameters
        ----------
        cells_to_process : array
            Indices to selected cells for data analysis. (Default) All ROIs identified in suite2p

        Returns
        -------
        None.

        """
        # if cells_to_process is defined in input, use the defined value
        if cells_to_process is not None:
            # check if cells to be processed is within the limits of stat
            arr1 = set(range(len(self.stat)))
            arr2 = set(cells_to_process)
            if not arr1.union(arr2) == arr1:  # not a subset
                print(_BColours.WARNING,
                      "object.cells_to_process is not a subset of the recognised ROIs. Resetting to the default "
                      "selection.", _BColours.ENDC)
                self.default_cells_to_process()
            else:
                self.cells_to_process = np.array(cells_to_process)
        else:
            self.default_cells_to_process()

    def check_cells_to_plot(self, cells_to_plot):
        """
        Check that the newly defined cells_to_plot is within the scope of cells_to_process

        Parameters
        ----------
        cells_to_plot : array
            Indices to selected cells for plotting. (Default) All ROIs within cells_to_process that are identified as
            cells in suite2p

        Returns
        -------
        None.

        """
        # if cells_to_plot is defined in input, use the defined value
        if cells_to_plot is not None:
            # check if cells to be plotted is a subset of cells to be processed
            arr1 = set(self.cells_to_process)
            arr2 = set(cells_to_plot)
            if not arr1.union(arr2) == arr1:  # not a subset
                print(_BColours.WARNING,
                      "object.cells_to_plot is not a subset of object.cells_to_process. Resetting to the default "
                      "selection.", _BColours.ENDC)
                self.default_cells_to_plot()
            else:
                self.cells_to_plot = cells_to_plot
        else:
            self.default_cells_to_plot()

    def default_cells_to_process(self):
        """
        Defining the default cells_to_process, which is all ROIs.

        Parameters
        ----------
        None.

        Returns
        -------
        None.

        """
        if self.stat is not None:  # by default selected cells for processing are all ROIs
            self.cells_to_process = np.array(range(len(self.stat)))
        else:
            self.cells_to_process = None
            print(_BColours.WARNING, "object.stat is not present. Please define object.cells_to_process for further "
                                     "processing and plotting.", _BColours.ENDC)

    def default_cells_to_plot(self):
        """
        Defining the default cells_to_plot, which is all real cells within cells_to_process.

        Parameters
        ----------
        None.

        Returns
        -------
        None.

        """
        if self.iscell is not None:  # by default selected cells for plotting are all real cells within cells_to_process
            k = np.nonzero(self.iscell[:, 0])
            k = np.array(k)
            k = k.reshape(-1)
            arr1 = set(self.cells_to_process)
            arr2 = set(k)
            self.cells_to_plot = np.array(list(arr1.intersection(arr2)))
        else:
            self.cells_to_plot = None
            print(_BColours.WARNING, "object.iscell is not present. Please define object.cells_to_plot for further "
                                     "processing and plotting.", _BColours.ENDC)

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
        t = _Timer(self._verbose)

        # calculate delta F over F
        if (self.F is not None) & (self.Fneu is not None):
            t.tic("Calculating delta F over F...")
            data = (self.F - self.Fneu) / self.Fneu
            t.toc()  # print elapsed time
        else:
            data = None
            print(_BColours.WARNING, "F and/or Fneu does not exist. Cannot calculate delta F over F.", _BColours.ENDC)

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
        t = _Timer(self._verbose)
        t.tic('Creating dataframe from suite2p stat...')

        if (self.stat is not None) and (self.iscell is not None):
            def get_data_from_stat(num_cells, item, col_name):
                data = np.zeros(num_cells)
                for n in range(0, num_cells):
                    data[n] = self.stat[n][item]
                self.ori_metadata[col_name] = data

            self.ori_metadata['iscell'] = self.iscell[:, 0]

            ncells = len(self.stat)  # include all the ROIs detected by suite2p
            get_data_from_stat(ncells, 'npix', 'area')
            get_data_from_stat(ncells, 'radius', 'radius')
            get_data_from_stat(ncells, 'aspect_ratio', 'aspect_ratio')
            get_data_from_stat(ncells, 'compact', 'compact')
            get_data_from_stat(ncells, 'solidity', 'solidity')

            self.ori_metadata['skip'] = np.zeros(ncells)

            t.toc()  # print elapsed time

        else:
            print(_BColours.WARNING, "object.stat and/or object.iscell does not exist. Cannot create dataframe from "
                                     "suite2p stat.", _BColours.ENDC)

    def print_ori_metadata(self):
        """
        Print information about the metadata obtained from suite2p.

        Parameters
        ----------
        None.

        Returns
        -------
        None.
        """
        plus_minus = u"\u00B1"
        print(_BColours.OKGREEN, 'Total number of ROIs = ', len(self.stat), _BColours.ENDC)
        print(_BColours.OKGREEN, 'Total number of ROIs classified as cells = ', int(np.sum(self.iscell, axis=0)[0]),
              _BColours.ENDC)

        # create table for displaying metadata
        table = Table(title="Metadata from Suite2p (Mean " + plus_minus + " SD)")

        table.add_column("Type", justify="right")
        table.add_column("Area\n(sq. px)", justify="center")
        table.add_column("Radius (px)", justify="center")
        table.add_column("Aspect Ratio", justify="center")
        table.add_column("Compact", justify="center")
        table.add_column("Solidity", justify="center")

        area = np.multiply(self.iscell[:, 0], self.ori_metadata.area.values[:])
        radius = np.multiply(self.iscell[:, 0], self.ori_metadata.radius.values[:])
        aspect_ratio = np.multiply(self.iscell[:, 0], self.ori_metadata.aspect_ratio.values[:])
        compact = np.multiply(self.iscell[:, 0], self.ori_metadata.compact.values[:])
        solidity = np.multiply(self.iscell[:, 0], self.ori_metadata.solidity.values[:])

        table.add_row("Cell",
                      f"{np.mean(area):0.1f} " + plus_minus + f" {np.std(area):0.1f}",
                      f"{np.mean(radius):0.1f} " + plus_minus + f" {np.std(radius):0.1f}",
                      f"{np.mean(aspect_ratio):0.1f} " + plus_minus + f" {np.std(aspect_ratio):0.1f}",
                      f"{np.mean(compact):0.1f} " + plus_minus + f" {np.std(compact):0.1f}",
                      f"{np.mean(solidity):0.1f} " + plus_minus + f" {np.std(solidity):0.1f}")

        area = np.multiply(-(self.iscell[:, 0] - 1), self.ori_metadata.area.values[:])
        radius = np.multiply(-(self.iscell[:, 0] - 1), self.ori_metadata.radius.values[:])
        aspect_ratio = np.multiply(-(self.iscell[:, 0] - 1), self.ori_metadata.aspect_ratio.values[:])
        compact = np.multiply(-(self.iscell[:, 0] - 1), self.ori_metadata.compact.values[:])
        solidity = np.multiply(-(self.iscell[:, 0] - 1), self.ori_metadata.solidity.values[:])

        table.add_row("Not Cell",
                      f"{np.mean(area):0.1f} " + plus_minus + f" {np.std(area):0.1f}",
                      f"{np.mean(radius):0.1f} " + plus_minus + f" {np.std(radius):0.1f}",
                      f"{np.mean(aspect_ratio):0.1f} " + plus_minus + f" {np.std(aspect_ratio):0.1f}",
                      f"{np.mean(compact):0.1f} " + plus_minus + f" {np.std(compact):0.1f}",
                      f"{np.mean(solidity):0.1f} " + plus_minus + f" {np.std(solidity):0.1f}")

        console = Console()
        console.print(table)

    def create_metadata(self, _print=True):
        """
        Calculate metadata of selected cells, columns include 'ROInum', 'iscell', 'area',
        'centroid', 'major_axis', 'minor_axis', 'orientation', 'aspect_ratio', 'circularity', 'perimeter', 'compact',
         'solidity', 'dfof'

        Parameters
        ----------
        _print : bool
            (FOR INTERNAL USE) Whether to print information about metadata after processing. (Default) True

        Returns
        -------
        None.

        """
        # start timer
        t = _Timer(self._verbose)
        t.tic('Calculating metadata...')

        # if metadata has not been defined before
        df = self.metadata
        if df.empty:

            self.metadata = self.init_df(self.cells_to_process)
            label_mask = np.zeros((self.ops['Ly'], self.ops['Lx']))
            label_mask = self.get_label_mask(label_mask, self.cells_to_process)
            self.label_mask = self.fill_df(label_mask)
            contour_mask = find_boundaries(self.label_mask, connectivity=1, mode='inner', background=0)
            self.contour_mask = np.multiply(contour_mask, label_mask)

        else:  # self.metadata has been defined before
            # compare new cell_to_process with current set
            old_arr = np.unique(self.label_mask) - 1
            old_arr = old_arr[1:]  # exclude value 0
            new_arr = self.cells_to_process

            # new_arr not in old_arr
            # -> add new cells to list
            idx = new_arr[np.isin(new_arr, old_arr, assume_unique=True, invert=True)]
            if idx.size != 0:
                label_mask = self.get_label_mask(self.label_mask, idx)
                self.metadata = pd.concat([self.metadata, self.init_df(idx)], ignore_index=True)
                self.metadata.sort_values(by=['ROInum'])
                label_mask = self.fill_df(label_mask)
                self.label_mask = label_mask + self.label_mask
                contour_mask = find_boundaries(label_mask, connectivity=1, mode='inner', background=0)
                self.contour_mask = np.multiply(contour_mask, label_mask) + self.contour_mask

            # old_arr not in new_arr
            # -> remove cell from dataframe, label_mask and contour_mask
            idx = old_arr[np.isin(old_arr, new_arr, assume_unique=True, invert=True)]
            if idx.size != 0:
                self.metadata = self.metadata[~self.metadata['ROInum'].isin(idx)]
                self.label_mask[np.isin(self.label_mask, idx + 1)] = 0
                self.contour_mask[np.isin(self.contour_mask, idx + 1)] = 0

        t.toc()  # print elapsed time
        if _print:
            self.print_metadata()

    def init_df(self, idx_list):
        # define columns for storing parameters to be calculated in later sessions
        max_dfof = np.max(self.dfof, axis=1)
        d = {'ROInum': idx_list, 'dfof': max_dfof[idx_list],
             'area': None, 'perimeter': None, 'centroid': None, 'orientation': None,
             'major_axis': None, 'minor_axis': None, 'aspect_ratio': None, 'circularity': None, 'compact': None,
             'solidity': None, 'iscell': None}
        df = pd.DataFrame(data=d)
        # this line allows the assignment of the array
        df = df.astype(object)
        return df

    def fill_df(self, label_mask):
        # get properties of the region and store in metadata
        regions = regionprops(label_mask.astype('int'))
        t1 = _Timer(verbose=True)
        t1.tic("region props")

        multiple_contour = [d for d in regions if d['euler_number'] > 1]

        # if any fo the regions has multiple contours, remove the smaller contour
        if multiple_contour:
            for props in multiple_contour:
                idx = self.metadata.index[self.metadata['ROInum'] == props.label - 1]  # the index to the ROI in metadata
                n = idx.values[0]
                contours = measure.find_contours(label_mask == self.metadata['ROInum'][n] + 1)
                lengths = [len(arr) for arr in contours]
                contours = contours[lengths.index(max(lengths))]

                label_mask1 = np.zeros((self.ops['Ly'], self.ops['Lx']))
                label_mask1[label_mask == props.label] = 1
                points = np.transpose(np.array(label_mask1.nonzero()))
                mask = measure.points_in_poly(points, contours)
                label_mask1[points[mask][:, 0], points[mask][:, 1]] = props.label
                label_mask1[points[~mask][:, 0], points[~mask][:, 1]] = 0

                # update label_mask
                label_mask[points[~mask][:, 0], points[~mask][:, 1]] = 0
                regions1 = regionprops(label_mask1.astype('int'))
                idx = next((index for (index, d) in enumerate(regions) if d["label"] == props.label), None)
                regions[idx] = regions1[0]

        # assign values to self.matadata
        idx = np.array([props['label'] for props in regions]) - 1
        idx = self.metadata.index[np.isin(self.metadata['ROInum'], idx)]
        n = idx.values[:]
        self.metadata['iscell'][n] = self.iscell[self.cells_to_process[n], 0]

        self.metadata['area'][n] = np.array([props['num_pixels'] for props in regions])  # number of pixels
        self.metadata['major_axis'][n] = np.array([props['axis_major_length'] for props in regions])
        self.metadata['minor_axis'][n] = np.array([props['axis_minor_length'] for props in regions])
        self.metadata['orientation'][n] = np.array([props['orientation'] for props in regions])
        self.metadata['perimeter'][n] = np.array([props['perimeter'] for props in regions])
        self.metadata['solidity'][n] = np.array([props['solidity'] for props in regions])
        self.metadata['centroid'][n] = [props['centroid'] for props in regions]

        t1.toc()
        # Trimming metadata to remove zero entries
        # area of cell has to be more than 1 pixel for calculation of major and minor axis
        area_threshold = 1
        df = self.metadata[self.metadata['area'].isnull()]
        idx = np.array(df['ROInum'], dtype='int')
        old_val = np.array(label_mask, dtype='int')
        label_mask[np.isin(old_val, idx + 1)] = 0
        self.ori_metadata['skip'][idx] = 1

        self.metadata = self.metadata.loc[self.metadata['area'] > area_threshold]
        self.cells_to_process = np.array(self.metadata['ROInum']).astype('int')

        # calculations for other items in metadata
        # ref: https://imagej.nih.gov/ij/docs/guide/146-30.html
        # self.metadata['aspect_ratio'] = self.metadata['major_axis'] / self.metadata['minor_axis']
        self.metadata['circularity'] = 4 * np.pi * np.divide(self.metadata['area'],
                                                             np.power(self.metadata['perimeter'], 2))
        self.metadata['compact'] = 4 / np.pi * np.divide(self.metadata['area'],
                                                         np.power(self.metadata['major_axis'], 2))

        return label_mask

    def get_label_mask(self, org_label_mask, idx_list):
        """
        Print information about the metadata calculated by PecanPie.

        Parameters
        ----------
        org_label_mask : ndarray
            original label_mask

        idx_list : array
            indexes of cells to be added to label_mask

        Returns
        -------
        None.
        """
        label_mask = np.zeros((self.ops['Ly'], self.ops['Lx']), dtype='int')
        # remove indexes if the cells have been computed before and area is None
        idx = np.nonzero(np.array(self.ori_metadata['skip']))
        idx = idx[0]
        idx_list = idx_list[np.isin(idx_list, idx, invert=True)]

        for n, m in enumerate(idx_list):
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

            # TODO: check that there is only one cell

            label_mask = label_mask + im * (m + 1)  # +1 such that the first element is not labeled as 0

            # pixels with overlapping cells are set to zero.
            im = np.where(im, label_mask, 0)
            label_mask = np.where(im > m + 1, 0, label_mask)

        org_label_mask = org_label_mask.astype('int')

        label_mask[np.multiply(label_mask, org_label_mask)] = 0

        return label_mask

    def print_metadata(self):
        """
        Print information about the metadata calculated by PecanPie.

        Parameters
        ----------
        None.

        Returns
        -------
        None.
        """
        plus_minus = u"\u00B1"
        print(_BColours.OKGREEN, 'Total number of ROIs = ', len(self.stat), _BColours.ENDC)
        print(_BColours.OKGREEN, 'Total number of ROIs selected for processing = ', len(self.cells_to_process),
              _BColours.ENDC)

        # create table for displaying metadata
        table = Table(title="Metadata of PecanPie (Mean " + plus_minus + " SD)")

        table.add_column("Type", justify="right")
        table.add_column("Area\n(sq. px)", justify="center")
        table.add_column("Major Axis (px)", justify="center")
        table.add_column("Aspect Ratio", justify="center")
        # table.add_column("Orientation (rad)", justify="center")
        # table.add_column("Circularity", justify="center")
        table.add_column("Compact", justify="center")
        table.add_column("Solidity", justify="center")

        area = np.multiply(self.metadata.iscell[:], self.metadata.area.values[:])
        major_axis = np.multiply(self.metadata.iscell[:], self.metadata.major_axis.values[:])
        aspect_ratio = np.multiply(self.metadata.iscell[:], self.metadata.aspect_ratio.values[:])
        # orientation = np.multiply(self.metadata.iscell[:], self.metadata.orientation.values[:])
        # circularity = np.multiply(self.metadata.iscell[:], self.metadata.circularity.values[:])
        compact = np.multiply(self.metadata.iscell[:], self.metadata.compact.values[:])
        solidity = np.multiply(self.metadata.iscell[:], self.metadata.solidity.values[:])

        table.add_row("Cell",
                      f"{np.mean(area):0.1f} " + plus_minus + f" {np.std(area):0.1f}",
                      f"{np.mean(major_axis):0.1f} " + plus_minus + f" {np.std(major_axis):0.1f}",
                      f"{np.mean(aspect_ratio):0.1f} " + plus_minus + f" {np.std(aspect_ratio):0.1f}",
                      # f"{np.mean(orientation):0.1f} " + plus_minus + f" {np.std(orientation):0.1f}",
                      # f"{np.mean(circularity):0.1f} " + plus_minus + f" {np.std(circularity):0.1f}",
                      f"{np.mean(compact):0.1f} " + plus_minus + f" {np.std(compact):0.1f}",
                      f"{np.mean(solidity):0.1f} " + plus_minus + f" {np.std(solidity):0.1f}")

        area = np.multiply(-(self.metadata.iscell[:] - 1), self.metadata.area.values[:])
        major_axis = np.multiply(-(self.metadata.iscell[:] - 1), self.metadata.major_axis.values[:])
        aspect_ratio = np.multiply(-(self.metadata.iscell[:] - 1), self.metadata.aspect_ratio.values[:])
        # orientation = np.multiply(-(self.metadata.iscell[:] - 1), self.metadata.orientation.values[:])
        # circularity = np.multiply(-(self.metadata.iscell[:] - 1), self.metadata.circularity.values[:])
        compact = np.multiply(-(self.metadata.iscell[:] - 1), self.metadata.compact.values[:])
        solidity = np.multiply(-(self.metadata.iscell[:] - 1), self.metadata.solidity.values[:])

        table.add_row("Not Cell",
                      f"{np.mean(area):0.1f} " + plus_minus + f" {np.std(area):0.1f}",
                      f"{np.mean(major_axis):0.1f} " + plus_minus + f" {np.std(major_axis):0.1f}",
                      f"{np.mean(aspect_ratio):0.1f} " + plus_minus + f" {np.std(aspect_ratio):0.1f}",
                      # f"{np.mean(orientation):0.1f} " + plus_minus + f" {np.std(orientation):0.1f}",
                      # f"{np.mean(circularity):0.1f} " + plus_minus + f" {np.std(circularity):0.1f}",
                      f"{np.mean(compact):0.1f} " + plus_minus + f" {np.std(compact):0.1f}",
                      f"{np.mean(solidity):0.1f} " + plus_minus + f" {np.std(solidity):0.1f}")

        console = Console()
        console.print(table)

    def change_cell_selection(self, cells_to_process=None, cells_to_plot=None):
        """
        Changing selections according to the array input.

        Parameters
        ----------
        cells_to_process : array
            Indices to selected cells for data analysis. (Default) All ROIs identified in suite2p

        cells_to_plot : array
            Indices to selected cells for plotting. (Default) All ROIs within cells_to_process that are identified as
            cells in suite2p

        Returns
        -------
        None.

        """
        self.check_cells_to_process(cells_to_process)
        self.check_cells_to_plot(cells_to_plot)
        self.create_metadata()

    # ------------------------------------------------------------------#
    #                           data analysis                           #
    # ------------------------------------------------------------------#

    # ------------------------------------------------------------------#
    #                          data visualisation                       #
    # ------------------------------------------------------------------#
    def create_fig(self, plottype, plot=True, filename=None):
        # TODO: add other parameters for more flexible plot
        """
        Sets parameters for plotting according to plot type.

        Parameters
        ----------
        plottype : str
            'avg_bin' = plots the registered binary data averaged over time
            'selected_cells' = plots the selected cells in peak delta F over F intensity
            'contour' = plots the selected cells with their contours after morphological operations
            'axis' = plots the selected cells with their contours and axes after morphological operations
            'cell_selection' = (FOR INTERNAL USE) for internactive selection of cells. Plots all cells with green
                               contour. Contours of cells not in self._tmp would be invisible.

        plot : bool
            Whether to show plot or not. (Default) True

        filename : str
            Filename of figure to save. If filename is not set, the figure will NOT be saved. (Default) None

        Returns
        -------
        None.

        """

        t = _Timer(self._verbose)
        t.tic("Creating plot for " + plottype + "...")

        # check the current number of plots
        if not self._im[0]:  # the first dictionary is empty
            k = 0
        else:  # the first dictionary is NOT empty. Append to the list of dictionaries
            k = len(self._im)
            self._im.append({})

        # for plotting of the average binary image
        if plottype == 'avg_bin':
            if self.bindata is not None:
                self._im[k]['imdata'] = np.mean(self.bindata, axis=0)
                self._im[k]['title'] = "Motion-Corrected Image"
                self._im[k]['cmap'] = 'gray'
                self._im[k]['type'] = 'image'

            else:
                print("data.bin does not exist. Plot type is not supported.")

        # for plotting ROIs in peak delta F over F intensity, the mask for ROI is from output of suite2p
        elif plottype == 'selected_cells':
            self._im[k]['imdata'] = self.switch_idx_to_value(self.metadata['dfof'])
            self._im[k]['title'] = "Selected cells at peak intensity"
            self._im[k]['cmap'] = 'gray'
            self._im[k]['type'] = 'image'

        # for plotting ROIs with their contours after morphological operations
        elif plottype == 'contour':
            self._im[k]['overlay'] = np.isin(self.contour_mask, self.cells_to_plot + 1)
            self._im[k]['imdata'] = self.switch_idx_to_value(self.metadata['dfof'])
            self._im[k]['title'] = "Contours of selected cells"
            self._im[k]['cmap'] = 'gray'
            self._im[k]['type'] = 'image'

        # for plotting ROIs with their contours and axes after morphological operations
        elif plottype == 'axis':
            self._im[k]['line_data'] = []

            for n in self.cells_to_plot:
                idx = self.metadata.index[self.metadata['ROInum'] == n]  # the index to the ROI in metadata

                # get the contour of ROI
                contour = self.metadata.loc[idx]['contour'].values[0]
                self._im[k]['line_data'].append({'x': contour[:, 1], 'y': contour[:, 0]})

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

                self._im[k]['line_data'].append(
                    {'x': (x0, x1), 'y': (y0, y1), 'color': self._color['red'], 'linewidth': self._linewidth})
                self._im[k]['line_data'].append(
                    {'x': (x0, x2), 'y': (y0, y2), 'color': self._color['red'], 'linewidth': self._linewidth})
            self._im[k]['imdata'] = self.switch_idx_to_value(self.metadata['dfof'])
            self._im[k]['title'] = "Axis of fitted ellipse"
            self._im[k]['cmap'] = 'gray'
            self._im[k]['type'] = 'image & line'

        # plotting all ROIs from stat for cell selection
        elif plottype == 'cell_selection':
            test_elements = np.array(self._tmp)
            self._im[k]['overlay'] = np.isin(self.contour_mask, test_elements+1)
            self._im[k]['imdata'] = self.switch_idx_to_value(self.metadata['dfof'])
            self._im[k]['title'] = "Click on cells to select or de-select, press ENTER to quit"
            self._im[k]['cmap'] = 'gray'
            self._im[k]['type'] = 'image'

        else:
            print(_BColours.WARNING, "Plot type is undefined.", _BColours.ENDC)
            return 0

        # determining the canvas
        if k == 0:  # the first plot
            self._im[k]['canvas'] = 0
        elif self._im[k - 1]['plot'] == 0:  # the previous plot is not shown
            self._im[k]['canvas'] = self._im[k - 1]['canvas']
        else:
            self._im[k]['canvas'] = self._im[k - 1]['canvas'] + 1

        self._im[k]['xlabel'] = '(pixels)'
        self._im[k]['ylabel'] = '(pixels)'
        self._im[k]['xlim'] = [0, self._im[k]['imdata'].shape[1] - 1]
        self._im[k]['ylim'] = [0, self._im[k]['imdata'].shape[0] - 1]

        self._im[k]['plot'] = plot
        self._im[k]['filename'] = filename

        t.toc()

    def switch_idx_to_value(self, value):
        """
        Switch index in label_mask to max dfof if index belongs to cells_to_plot
        Switch index in label_mask to 0 if index belongs to cells_to_process but not cells_to_plot

        Parameters
        ----------
        None.

        Returns
        -------
        None.

        """

        a = self.label_mask.astype('int')
        a = a.reshape((self.label_mask.size,))
        val_old = np.array(self.metadata['ROInum'].astype('int') + 1)
        val_new = np.array(value)

        arr = np.empty(a.max() + 1, dtype='float')
        arr[val_old] = val_new
        a = arr[a]
        a[a==None] = 0  # DO NOT change to "a is None"
        a = a.reshape((self.label_mask.shape[0], self.label_mask.shape[1]))
        a = np.array(a, dtype='float')
        return a

    def plot_fig(self, _ion=False):
        """
        Visualize image data and saving

        Parameters
        ----------
        _ion : bool
            (FOR INTERNAL USE) Whether to turn interactive mode on. (Default) False

        Returns
        -------
        None.

        """
        for k in range(0, len(self._im)):
            if self._im[k]['plot'] is True:
                plt.figure(self._im[k]['canvas'])
                plt.title(self._im[k]['title'])
                plt.xlabel(self._im[k]['xlabel'])
                plt.ylabel(self._im[k]['ylabel'])
                plt.xlim(self._im[k]['xlim'])
                plt.ylim(self._im[k]['ylim'])

                if self._im[k]['type'] == 'image':
                    plt.imshow(self._im[k]['imdata'], cmap=self._im[k]['cmap'])
                    if self._im[k]['overlay'] is not None:
                        self.overlay_handle = plt.imshow(self._im[k]['overlay'], cmap='pp_cmap')

                elif self._im[k]['type'] == 'image & line':
                    plt.imshow(self._im[k]['imdata'], cmap=self._im[k]['cmap'])

                    num_lines = len(self._im[k]['line_data'])
                    for n in range(num_lines):
                        x = self._im[k]['line_data'][n]['x']
                        y = self._im[k]['line_data'][n]['y']

                        if 'color' in self._im[k]['line_data'][n]:
                            color = self._im[k]['line_data'][n]['color']
                        else:
                            color = None

                        if 'linewidth' in self._im[k]['line_data'][n]:
                            linewidth = self._im[k]['line_data'][n]['linewidth']
                        else:
                            linewidth = None

                        if color is None:
                            plt.plot(x, y, linewidth=linewidth)
                        else:
                            plt.plot(x, y, color=color, linewidth=linewidth)

                        if 'visible' in self._im[k]['line_data'][n]:
                            visible = self._im[k]['line_data'][n]['visible']
                            ax = plt.gca()
                            plt.setp(ax.lines[n], visible=visible)

            if self._im[k]['filename'] is not None:  # save as a tif file if 'filename' has been assigned
                if not self._im[k]['filename'].endswith(".tif"):
                    self._im[k]['filename'] = self._im[k]['filename'] + ".tif"
                plt.savefig(self.save_path_fig + self._im[k]['filename'])

        if _ion:
            plt.ion()
            plt.show()
        else:
            plt.show(block=True)
        self._im = [{}]  # clear list after plotting

    # ------------------------------------------------------------------#
    #                           Data Exploration                        #
    # ------------------------------------------------------------------#
    def cells_to_process_from_fig(self):
        """
        Open an interactive graphical interface for selecting and deselecting cells in cells_to_process

        Parameters
        ----------
        None.

        Returns
        -------
        None.

        """
        self._tmp = np.array(self.cells_to_process)
        tmp_selection_plot = self.cells_to_plot

        self.default_cells_to_process()  # reset selection to default

        self.create_metadata(_print=False)
        self.cells_to_plot = self.cells_to_process
        self.create_fig('cell_selection', plot=True)
        self.plot_fig(_ion=True)

        self.cells_to_process = self.get_selection()
        self.cells_to_plot = np.array(list(set(self.cells_to_process).intersection(tmp_selection_plot)))
        self.create_metadata()

        self._tmp = []

    def cells_to_plot_from_fig(self):
        """
        Open an interactive graphical interface for selecting and deselecting cells in cells_to_plot

        Parameters
        ----------
        None.

        Returns
        -------
        None.

        """
        self._tmp = np.array(self.cells_to_plot)
        self.cells_to_plot = self.cells_to_process
        self.create_fig('cell_selection', plot=True)
        self.plot_fig(_ion=True)
        self.cells_to_plot = self.get_selection()
        self._tmp = []
        print(_BColours.OKGREEN, 'Total number of ROIs selected for processing = ', len(self.cells_to_process),
              _BColours.ENDC)
        print(_BColours.OKGREEN, 'Total number of ROIs selected for plotting = ', len(self.cells_to_plot),
              _BColours.ENDC)

    def get_selection(self):
        """
        Get point from graph and update the temporary selection

        Parameters
        ----------
        None.

        Returns
        -------
        tmp_selection : array
            temporary selection of cells

        """
        ax = plt.gca()
        tmp_selection = self._tmp

        # display ROInum while cursor hover on cell
        cursor = mplcursors.cursor(hover=mplcursors.HoverMode.Transient)

        @cursor.connect("add")
        def on_add(sel):
            cursor_x, cursor_y = sel.target
            cursor_x = round(cursor_x)
            cursor_y = round(cursor_y)
            ROInum = int(self.label_mask[cursor_y, cursor_x] - 1)  # -1 to convert label to ROInum
            if ROInum != -1:
                sel.annotation.set(text=f"ROI: {ROInum}",
                                   bbox=dict(boxstyle=None, fc="lightblue", ec="lightblue", alpha=0.5))
            else:
                sel.annotation.set(text=None)

        while True:
            try:

                pts = np.rint(np.array(plt.ginput(1, timeout=-1)))  # timeout = -1 for no timeout
                x = int(pts[0, 0])
                y = int(pts[0, 1])
                ROInum = int(self.label_mask[y, x] - 1)  # -1 to convert label to ROInum

                if ROInum in tmp_selection:  # remove from selection
                    tmp_selection = np.delete(tmp_selection, np.where(tmp_selection == ROInum))
                    self.overlay_handle.remove()
                    test_elements = np.array(tmp_selection)
                    self.overlay_handle = plt.imshow(np.isin(self.contour_mask, test_elements + 1, invert=False),
                                                     cmap='pp_cmap')

                elif ROInum in self.cells_to_plot:  # add to selection
                    tmp_selection = np.sort(np.append(tmp_selection, ROInum))
                    t = _Timer(self._verbose)
                    t.tic("remap contour")
                    self.overlay_handle.remove()
                    test_elements = np.array(tmp_selection)
                    self.overlay_handle = plt.imshow(np.isin(self.contour_mask, test_elements + 1, invert=False),
                                                     cmap='pp_cmap')

                else:  # case where ROInum is not recognized, meaning a click on the empty parts of the plot.
                    continue

            except IndexError or TypeError:
                # If no pts is read from ginput, IndexError would occur at "x = pts[0, 0].astype('int')"
                plt.ioff()
                plt.close()
                break

        return tmp_selection


# ------------------------------------------------------------------#
#                               Timer                               #
# ------------------------------------------------------------------#


class _Timer:
    """
    Timer for checking performance

    Parameters
    ----------
    verbose : bool
        Whether to print timing in console or not. (Default) False

    txt : str
        Description of the current process.

    Returns
    -------
    None.

    """

    def __init__(self, verbose=False):
        self._verbose = verbose
        self._start_time = time.perf_counter()
        self._len_of_txt = 0
        self._txt = None

    def tic(self, txt):
        # Start a new timer
        self._start_time = time.perf_counter()
        self._txt = txt
        self._len_of_txt = len(txt)

    def toc(self):
        console_width = 80
        # Stop the timer, and report the elapsed time
        elapsed_time = time.perf_counter() - self._start_time
        if self._verbose:
            status = f"Elapsed time: {elapsed_time:0.4f} seconds"
            status = status.rjust(console_width - self._len_of_txt, ".")
            print(f"{self._txt}{status}")

        self._start_time = time.perf_counter()


# ------------------------------------------------------------------#
#                     color of console print                        #
# ------------------------------------------------------------------#

class _BColours:
    """
    Colours for printing

    Parameters
    ----------
    None.

    Returns
    -------
    None.

    """
    HEADER = '\033[95m'  # pink
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'  # yellow
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


# ------------------------------------------------------------------#
#                       colormap for contour                        #
# ------------------------------------------------------------------#
def create_cmap(color):
    """
        Colourmap for overlaying contour on intensity

        Parameters
        ----------
        color : color of contour for selected cells

        Returns
        -------
        None.

        """
    color = np.array([mpl.colors.to_rgba(color)])
    trans_color = np.array([[1, 1, 1, 0]])
    color_array = np.concatenate((trans_color, color), axis=0)

    # create a colormap object
    map_object = LinearSegmentedColormap.from_list(name='pp_cmap', colors=color_array)

    # register this new colormap with matplotlib
    mpl.colormaps.register(cmap=map_object)


# ------------------------------------------------------------------#
#                         static methods                            #
# ------------------------------------------------------------------#
def print_data_status(val, txt):
    """
    Print information about a parameter.

    Parameters
    ----------
    val : number / bool
        Value of the parameter. 1 (True) to print a tick. 0 (False) to print a cross. Other values would be printed
        as they are.

    txt : str
        Name of the parameter to print out.

    Returns
    -------
    None.

    """
    tick = u'\u2713'
    cross = 'X'
    console_width = 79
    txt = txt.split('.')[-1]

    if val == 1:
        status = tick.rjust(console_width - len(txt), ".")
    elif val == 0:
        status = cross.rjust(console_width - len(txt), ".")
    else:  # other numerical values
        status = str(val).rjust(console_width - len(txt), ".")

    print(f"{txt} {status}")
