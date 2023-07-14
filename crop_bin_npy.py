# This script crops .npy and .bin files
# Author                   : Jane Ling
# Date of creation         : 29/06/2023
# Date of last modification: 14/07/2023

# ------------------------------------------------------------------#
#                         load packages                             #
# ------------------------------------------------------------------#
import math
import numpy as np
import pecan_pie as pp

# ------------------------------------------------------------------#
#                     set paths and size of output                  #
# ------------------------------------------------------------------#
# set filepath to outputs folder
filepath = '/media/cdn-bc/RAID/Datasets/Harris_BioRxiv_2016_suite2p/outputs/suite2p/plane0/'

filepath_save = '/home/jane/PycharmProjects/Suite2p/testdata/originals/'

lenx = 90
leny = 100
startframe = 0
stopframe = 500


# ------------------------------------------------------------------#
#                            functions                              #
# ------------------------------------------------------------------#
def modify_stat(stat):
    stat = np.reshape(stat, (len(stat),))
    for k in range(0, len(stat)):
        stat[k]['ypix'] = stat[k]['ypix'] - starty
        stat[k]['xpix'] = stat[k]['xpix'] - startx
        stat[k]['med'][0] = stat[k]['med'][0] - starty
        stat[k]['med'][1] = stat[k]['med'][1] - startx

    return stat


# ------------------------------------------------------------------#
#                            load data                              #
# ------------------------------------------------------------------#
# load .npy bindata from read_path
s2p = pp.PecanPie(filepath)

# ------------------------------------------------------------------#
#                           modify bin                              #
# ------------------------------------------------------------------#

startx = math.ceil(s2p.bindata.shape[2]/2-lenx/2)
starty = math.ceil(s2p.bindata.shape[1]/2-leny/2)
stopx = startx + lenx
stopy = starty + leny


byte_list = s2p.bindata[startframe:stopframe, starty:stopy, startx:stopx]
size_bytelist = (stopframe-startframe)*lenx*leny
byte_list = np.reshape(byte_list, (size_bytelist,))
byte_array = bytes(byte_list)

# ------------------------------------------------------------------#
#                            save bin                               #
# ------------------------------------------------------------------#
try:
    with open(filepath_save + 'data.bin', 'wb') as f:
        f.write(byte_array)
        print("Done saving data.bin")
except Exception as e:
    print(e)

# ------------------------------------------------------------------#
#                           modify npy                              #
# ------------------------------------------------------------------#

# keep only parameters in ops necessary for running toolbox
# TODO: update list if there is a change to toolbox
ops = {'Ly': leny, 'Lx': lenx, 'xrange': [0, lenx - 1], 'yrange': [0, leny - 1], 'nframes': stopframe - startframe}

# ------------------------------------------------------------------#
#                           modify npy                              #
# ------------------------------------------------------------------#
ncells = len(s2p.iscell)
m = 0
for n in range(0, ncells):
    ypix = s2p.stat[n]['ypix'][:]
    xpix = s2p.stat[n]['xpix'][:]
    if any(ypix > stopy) or any(ypix < starty):  # y index of cell is out of range
        continue
    elif any(xpix > stopx) or any(xpix < startx):  # x index of cell is out of range
        continue
    elif m == 0:
        F_new = s2p.F[n][startframe:stopframe]
        Fneu_new = s2p.Fneu[n][startframe:stopframe]
        spks_new = s2p.spks[n][startframe:stopframe]
        stat_new = s2p.stat[n]
        iscell_new = s2p.iscell[n][:]
        m = m + 1
    else:
        F_new = np.row_stack((F_new, s2p.F[n][startframe:stopframe]))
        Fneu_new = np.row_stack((Fneu_new, s2p.Fneu[n][startframe:stopframe]))
        spks_new = np.row_stack((spks_new, s2p.spks[n][startframe:stopframe]))
        stat_new = np.row_stack((stat_new, s2p.stat[n]))
        iscell_new = np.row_stack((iscell_new, s2p.iscell[n][:]))
        m = m + 1

stat_new = modify_stat(stat_new)

# rename arrays
F = F_new
Fneu = Fneu_new
spks = spks_new
stat = stat_new
iscell = iscell_new

# ------------------------------------------------------------------#
#                            save npy                               #
# ------------------------------------------------------------------#
np.save(filepath_save+'F.npy', F)
np.save(filepath_save+'Fneu.npy', Fneu)
np.save(filepath_save+'spks.npy', spks)
np.save(filepath_save+'stat.npy', stat)
np.save(filepath_save+'ops.npy', ops)
np.save(filepath_save+'iscell.npy', iscell)


