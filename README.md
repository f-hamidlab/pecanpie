**suite2p_tools** <br>
**class s2p** <br>
**parameters** <br>
    
    # paths
    - self.read_path     : str, path to directory with data files (.npy and .bin)
    - self.save_path     : str, path to directory for saving. (Default) same as read_path
    - self.save_path_fig : str, path to directory for saving figure outputs
    - self.save_path_data: str, path to directory for saving data outputs

    # suite2p outputs
    - self.F             : ndarray (ROIs x timepoints), fluorescence traces (i.e. raw)
    - self.Fneu          : ndarray (ROIs x timepoints), neuropil fluorescence traces (i.e. baseline)
    - self.spks          : ndarray (ROIs x timepoints), deconvolved traces (i.e. activities)
    - self.stat          : list (ROIs x 1), statistics computed for each cell
    - self.ops           : dictionary, options and intermediate outputs
    - self.iscell        : ndarray (ROIs x 2), specifies whether an ROI is a cell, first column is 0/1, and second column is probability that the ROI is a cell based on the default classifier
    - self.bindata       : ndarray (time x Ny x Nx), registered (i.e. motion-corrected) images

    # image data for plot/saving
    - self.im            : list of dictionaries
                           ['data']     : ndarray, data to plot, (Ny x Nx) for image, (ROIs x timepoints) for line
                           ['title']    : str, figure title
                           ['xlabel']   : str, label of x-axis
                           ['ylabel']   : str, label of y-axis
                           ['xlim']     : array, [xmin, xmax]
                           ['ylim']     : array, [ymin, ymax]
                           ['cmap']     : str, colormap
                           ['canvas']   : int, number of the canvas to plot
                           ['type']     : str, type of plot, 'line', 'image', 'scatter'
                           ['plot']     : bool, whether to show plot or not
                           ['filename'] : str, filename of figure to save.

    # options and intermediate outputs
    - self.selected_cells: array of indexes to selected cells, (Default) list of cells identified in suite2p
    - self.dfof          : ndarray (ROIs x timepoints), delta F over F


**functions for external use** <br>
    
    - self.__init__(read_path, save_path=None)
        > initialize the class object s2p
        > INPUTS:
            - read_path : str
                path to directory with data files (.npy and .bin)
            - save_path : str
                path to directory for saving. If undefined, same as read_path.
        > OUTPUTS:
            Parameters listed above
        > RETURNS: None

    - self.im_plot(plottype, plot=True, filename=None)
        > Sets parameters for plotting according to plot type.
        > INPUTS:
            - plottype : str
                'avg_bin'    = plots the registered binary data averaged over time
                'real_cells' = plots the real cells in peak intensity
            - plot : bool
                Whether to show plot or not. (Default) True
            - filename : str
                Filename of figure to save. If filename is not set, the figure will NOT be saved. (Default) None
        > OUTPUTS: If filename is specified, figure saved as .tif. Otherwise, none.
        > RETURNS: None

    - self.plot_fig()
        > Visualize image data and saving
        > INPUTS: None
        > OUTPUTS: If filename is specified, figure saved as .tif. Otherwise, none.
        > RETURNS: None

**functions for internal use** <br>
    
    - self.read_npy(filename)
        > Loads data from .npy
        > INPUTS:
            - filename : str
                filename of the .npy data file
        > OUTPUTS: None
        > RETURNS:
            - data
                ndarray (ROIs x timepoints), data stored in the .npy data file

    - self.cal_dfof()
        > Calculates delta F over F.
        > INPUTS: None
        > OUTPUTS: None
        > RETURNS:
            - data
                ndarray (ROIs x timepoints), delta F over F


**class Timer** <br>
**functions**
    
    - self.__init__()
        > Initialize timer with current time
        > INPUTS: None
        > OUTPUTS: None
        > RETURNS: None
    - self.tic()
        > Start a new timer with current time
        > INPUTS: None
        > OUTPUTS: None
        > RETURNS: None
    - self.toc()
        > Ends the timer and prints elapsed time. Restart timer.
        > INPUTS: None
        > OUTPUTS: None
        > RETURNS: None
