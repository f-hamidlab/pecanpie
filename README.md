**suite2p_tools** <br>

# Installation

```bash
git clone git@github.com:f-hamidlab/suite2ptools.git
cd suite2ptools

conda create -n s2ptools python=3.8
conda activate s2ptools
pip install .

# or to install suite2ptools in editable mode
# pip install -e . 
```

# class s2p
## Parameters
### Paths
| Parameter            | Data Type | Dimensions  | Description                                                   |
| :------------------- | :-------- | :---------- | :------------------------------------------------------------ |
| self.read_path       | str       | N/A         | path to directory with data files (.npy and .bin)             |
| self.save_path       | str       | N/A         | path to directory for saving. <br>(Default) same as read_path |
| self.save_path_fig   | str       | N/A         | path to directory for saving figure outputs                   |
| self.save_path_data  | str       | N/A         | path to directory for saving data outputs                     |

### Outputs from suite2p
| Parameter            | Data Type  | Dimensions        | Description                                               |
| :------------------- | :--------- | :---------------- | :-------------------------------------------------------- |
| self.F               | ndarray    | ROIs x timepoints | fluorescence traces (i.e. raw)                            |
| self.Fneu            | ndarray    | ROIs x timepoints | neuropil fluorescence traces (i.e. baseline)              |
| self.spks            | ndarray    | ROIs x timepoints | deconvolved traces (i.e. activities)                      |
| self.stat            | list       | ROIs x 1          | statistics computed for each cell                         |
| self.ops             | dictionary | N/A               | options and intermediate outputs                          |
| self.iscell          | ndarray    | ROIs x 2          | specifies whether an ROI is a cell, first column is 0/1, and second column is probability that the ROI is a cell based on the default classifier |
| self.bindata         | ndarray    | time x Ny x Nx    | registered (i.e. motion-corrected) images                 |

### Image data for plot/saving
- self.im            : list of dictionaries
    - ['data']     : ndarray, data to plot, (Ny x Nx) for image, (ROIs x timepoints) for line
    - ['title']    : str, figure title
    - ['xlabel']   : str, label of x-axis
    - ['ylabel']   : str, label of y-axis
    - ['xlim']     : array, [xmin, xmax]
    - ['ylim']     : array, [ymin, ymax]
    - ['cmap']     : str, colormap
    - ['canvas']   : int, number of the canvas to plot
    - ['type']     : str, type of plot, 'line', 'image', 'scatter'
    - ['plot']     : bool, whether to show plot or not
    - ['filename'] : str, filename of figure to save.

### Options and intermediate outputs
| Parameter           | Data Type | Dimensions        | Description                                                                        |
|:--------------------|:----------|:------------------|:-----------------------------------------------------------------------------------|
| self.selected_cells | ndarray   | N x 1             | indexes to selected cells <br>(Default) list of cells identified in suite2p        |
| self.dfof           | ndarray   | ROIs x timepoints | delta F over F                                                                     |
| self.df             | DataFrame | ROIs x properties | columns include 'iscell', 'area', 'radius', 'aspect_ratio', 'compact', 'solidity'  |

## Functions for external use
- self.__init__(read_path, save_path=None)
    - Initializes the class object s2p
    - INPUTS:
        - read_path : str <br>
            path to directory with data files (.npy and .bin)
        - save_path : str <br>
            path to directory for saving. If undefined, same as read_path.
    - OUTPUTS: <br>
        Parameters listed above
    - RETURNS: None

- self.im_plot(plottype, plot=True, filename=None)
    - Sets parameters for plotting according to plot type.
    - INPUTS:
        - plottype : str <br>
            - 'avg_bin'    = plots the registered binary data averaged over time
            - 'selected_cells' = plots the selected cells in peak delta F over F intensity
        - plot : bool
            Whether to show plot or not. (Default) True
        - filename : str <br>
            Filename of figure to save. If filename is not set, the figure will NOT be saved. (Default) None
    - OUTPUTS: If filename is specified, figure saved as .tif. Otherwise, none.
    - RETURNS: None

- self.plot_fig()
    - Visualizes image data and saving
    - INPUTS: None
    - OUTPUTS: If filename is specified, figure saved as .tif. Otherwise, none.
    - RETURNS: None

## Functions for internal use
- self.create_dataframe()
    - Create a DataFrame for storing metadata of cells. Insert existing data from self.stat and self.iscell.
    - INPUTS: None
    - OUTPUTS:
        - self.df: DataFrame
    - RETURNS: None
- self.read_npy(filename)
    - Loads data from .npy
    - INPUTS:
        - filename : str <br>
            filename of the .npy data file
    - OUTPUTS: None
    - RETURNS:
        - data <br>
            ndarray (ROIs x timepoints), data stored in the .npy data file

- self.cal_dfof()
    - Calculates delta F over F.
    - INPUTS: None
    - OUTPUTS: None
    - RETURNS:
        - data
            ndarray (ROIs x timepoints), delta F over F


# class Timer
## Functions
- self.__init__()
    - Initializes timer with current time
    - INPUTS: None
    - OUTPUTS: None
    - RETURNS: None

- self.tic()
    - Starts a new timer with current time
    - INPUTS: None
    - OUTPUTS: None
    - RETURNS: None
    
- self.toc()
    - Ends the timer and prints elapsed time. Restart timer.
    - INPUTS: None
    - OUTPUTS: None
    - RETURNS: None
