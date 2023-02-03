"""
PlotDisPerSEfilaments.py

This python file owns the class for plotting the DisPcatalog outputted by the DisPerSE algorithm

Script written by: Soumya Shreeram
Date created: 27th October 2022
Contact: shreeram@mpe.mpg.de
"""

import numpy as np

from astropy.io import fits
from astropy.table import Table, Column, join
from astropy import units as u
from astropy import wcs
import glob

import os
import subprocess
import glob
import logging
from scipy.ndimage import gaussian_filter
from scipy.interpolate import griddata

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from astropy.visualization import simple_norm
from scipy.ndimage import gaussian_filter

import healpy as hp
from .StraightenFilaments import StraightenFilaments
from .utils import custom_colormap, set_labels, set_as_white


class PlotDisPerSEfilaments(StraightenFilaments):
    """Class to visualize the filament images and other data products generated
    """
    def __init__(self, data_set: str = 'lc_north_dis', smoothing_density_f: str = "None", persistence: float = 3, smoothing_skeleton: str = "None", section_keyword: str = 'fil', fil_id: int = None):
        """
        Initialized when an object of PlotDisPerSEfilaments is called

        Parameters
        ----------
        eRASS4_dir :: str
            directory where the eRASS:4 data is stored
        save_data_dir :: str
            directory to save the data products generatated as part of this class
        data_set :: str
            the dataset on which the skeleton has been detected 
            ('legacy_north_dis' for Legacy MGS, 'lc_north_dis' for LOWZ+CMASS)
        base_dir :: str
            the base directory
        smoothing_density_f :: str
            the smoothing of the density field (None or 'SD1' for 1 smoothing cycle ...)
        persistence :: int
            the persistence threshold ('3' or '5')
        smoothing_skeleton :: str
            the smoothing of the skeleton (None or 'S001' for 1 smoothing cycle ...)
        ra_c, dec_c :: float
            the filament algorithm is centers at (ra, dec), so when cross-matching with other catalogs this value must be subtracted
        """
        # initialized the interited class parameters
        super().__init__(data_set=data_set, smoothing_density_f=smoothing_density_f, persistence=persistence, smoothing_skeleton=smoothing_skeleton, fil_id=fil_id)

        logging.basicConfig(level = logging.INFO)
        self.logger = logging.getLogger(type(self).__name__)

        # matplotlib plotting params
        mpl.rcParams['agg.path.chunksize'] = 100000
        font = {'family' : 'serif',
        'weight' : 'medium',
        'size'   : 18}
        mpl.rc('font', **font)

        # get the pixel values in the filament sub-image
        self.get_rotated_filament_pixels()

        if fil_segment_id is None:
            fil_segment_id = 2

        # useful quatitites required for plottings 
        sub_image_filenames = glob.glob(f"{self.inputs_folder}/ima_0.2_2.3_*.fits.gz")
        cd_matrix_arr, _ = self.get_rotated_filament_pixels()


    def plot_filament_sub_image(self, black_backgroup: bool = True):
        
        fil_ends_og, _, input_data, _, cutout_img, _ = self.get_straighted_images()
        fil_ends_og = fil_ends_og.T

        fig = plt.figure(figsize=(7, 7))
        
        ax = plt.subplot(1,1,1)
        norm1 = simple_norm(input_data, 'log', min_cut=1e-2, max_cut=1)
        ax.imshow(input_data, origin='lower', cmap='afmhot', norm=norm1)
        ax.plot(fil_ends_og[:, 0], fil_ends_og[:, 3], origin='lower', cmap='afmhot', norm=norm1)
        
        fo.set_labels(ax, r"x [pixels]", 'y [pixels]', title='Original filament sub-image')
        
        if black_backgroup:
            fig.set_facecolor('black')
            fo.set_as_white(ax)
        return 

    def plot_straight_and_rotated_image(self):
        return 
        