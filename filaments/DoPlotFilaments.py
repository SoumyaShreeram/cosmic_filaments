#!/usr/bin/pythons

"""
This python file contains the class for DoPlotFilaments. 
This class aims to grapically represent the information generated from the class Filaments

Author: Soumya Shreeram
Email: shreeram@mpe.mpg.de
Date created: 08 - Nov - 2021
"""
import numpy as np
import pandas as pd

from astropy.io import fits
from astropy.table import Table, Column, join
#!/usr/bin/pythons

import os
import subprocess
import sys
import logging

# plotting imports
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import matplotlib.patches as patches
import seaborn as sns

import importlib as ib

from .Filaments import Filaments
from .utils import aitoffProjection


class DoPlotFilaments(Filaments):
    """
    Graphical representation of the information in class Filaments

    """
    def __init__(self, this_filament_obj: object = None, image_dir: str = None):
        """Initialiasis when an object of the class DoPlotFilaments is created

        Parameters
        ----------
        this_filament_obj: object of class Filaments
            the object defines the corresponding filament plots done to represent it
        image_dir : str
            path to the directory where the plots produced in this class are saved
        """
        logging.basicConfig(level = logging.INFO)
        self.logger = logging.getLogger(type(self).__name__)

        if this_filament_obj is None:
            raise Warning("need to pass an object belonging to the class 'Filaments'")
        else:
            self.this_filament_obj = this_filament_obj

        self.sns_colors = sns.color_palette("tab10")

        if image_dir is None:
            image_dir = '/data53s/shreeram/Filament_stacking/images/01_filament_cat_plots/'
        self.image_dir = image_dir
        

    def plot_filaments(self, ax, color: str ='r', title: bool = True,  ms: float = 0.03,\
    alpha: float = 0.1):
        """
        """
        fil_cat_name = self.this_filament_obj.filament_catalogue_name
        self.logger.info(f" Plotting {fil_cat_name} on the all sky projection.")
        fil_table, ra_fil, dec_fil = self.this_filament_obj.open_fil_cat()    
        
        ra_fil, dec_fil = aitoffProjection(ra_fil, dec_fil)
        
        # plot on all sky-map
        ax.plot(ra_fil, dec_fil, '.', c=color, alpha=0.1, ms=ms, label=fil_cat_name)

        # set title
        if title:
            ax.set_title(self.this_filament_obj.catalog_file_description+'\n')
        
        self.filament_cat_name = fil_cat_name
        return ax

    def plot_filaments_on_sky(self, color: str ='r', title: bool = True,  ms: float = 0.03, \
        alpha: float = 0.1):
        """Method plots the filament catalogue on the all sky in mollweide projection
        """
        fig = plt.figure(figsize=(14,6.2))
        ax = plt.subplot(111, projection="mollweide")

        colorblind_colors = sns.color_palette("colorblind")
        ax = self.plot_filaments(ax, color=color, ms=ms, title=title, alpha=alpha)
        ax.grid()

        l = ax.legend(bbox_to_anchor=(1., 1.05))
        for legend_handle in l.legendHandles:
            legend_handle._legmarker.set_markersize(12)
            legend_handle.set_alpha(1)
        fig.patch.set_facecolor('white')
        fig.savefig(f"{self.image_dir}/{self.this_filament_obj.filament_catalogue_name}.png", format='png')
        return ax 
        
    def plot_filament_lengths_Tempel14(self, ax, fil_length_cutoff: int = 5, len_bins: int = 50):
        """Method to plot the filament length distributions of the optical catalogues used in the study
        
        Parameters
        ----------
        len_bins: int; default set to 50
            int decides the number of bins in the plotted histogram
        """
        fil_cat_name = self.this_filament_obj.filament_catalogue_name
        self.logger.info(f" Plotting {fil_cat_name} on the all sky projection.")
        fil_table, ra_fil, dec_fil, fil_lengths = self.this_filament_obj.open_fil_cat(return_lengths=True)   
        
        # get the filaments in the eRASS sky    
        overlap = self.this_filament_obj.get_overlapping_filaments(ra_fil=ra_fil, dec_fil=dec_fil)
        if len(overlap) > 0:
            self.logger.info(" We have overlapping filaments with the eRASS sky!")
            fil_IDs = np.unique(fil_table[1][:, 0][overlap.astype(int)])
            
            fil_lengths_bool = np.in1d(fil_table[0][:, 0], fil_IDs)
            fil_lengths = fil_table[0][:, 2][fil_lengths_bool]
            self.logger.info(f" Number of filaments found is {len(fil_lengths)}")
        
        # eRASS sky tile edges
        ax.hist(fil_lengths, color='#bda155', bins=len_bins, histtype='stepfilled', label='All overlapping filaments')
        ax.hist(fil_lengths[fil_lengths>fil_length_cutoff], color='#44c77f', bins=len_bins, histtype='stepfilled', label='Filaments > {fil_length_cutoff} Mpc')
        self.logger.info(f" Number of filaments longer than {fil_length_cutoff} Mpc: {len(fil_lengths[fil_lengths>fil_length_cutoff])}")
        self.fil_length_cutoff = fil_length_cutoff

        ax.set_xlabel("Filament lengths [Mpc]")
        ax.set_ylabel("Number of filaments")
        ax.set_title(self.this_filament_obj.filament_catalogue_name)
        self.fil_length_cutoff = fil_length_cutoff
        return ax

    def plot_eRASS_sky_tiles(self, axis):
        "Mehtod to plot the eRASS sky tiles"
        eRASS_ra_min, eRASS_dec_min, eRASS_ra_max, eRASS_dec_max = self.this_filament_obj.readeRASSskyTileFile()
        
        for i in range(len(eRASS_ra_max)):
            area_polygon = np.abs(eRASS_ra_max[i] - eRASS_ra_min[i])*np.abs(eRASS_dec_max[i] - eRASS_dec_min[i])
            
            # number comes from a test to eliminate the "psuedo" sky tiles in the 360 deg proj
            if area_polygon < 0.1:
                polygon = patches.Polygon([[eRASS_ra_min[i], eRASS_dec_min[i]],\
                                           [eRASS_ra_min[i], eRASS_dec_max[i]],\
                                           [eRASS_ra_max[i], eRASS_dec_max[i]],\
                                           [eRASS_ra_max[i], eRASS_dec_min[i]]],\
                                          fc='#74b0aa', alpha=0.5, ec = '#344178', closed=False)
                axis.add_patch(polygon)
        return axis

    def plot_eRASS_sky(self, axis):
        sky_data = np.load('/data26s/mpecl/data.npy')


