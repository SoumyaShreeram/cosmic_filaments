"""
Filaments2Files.py

This python file owns the classs to process filaments to produce:
images, exposures, masks.

Script written by: Soumya Shreeram
Date created: 1st April 2022
Contact: shreeram@mpe.mpg.de
"""
import numpy as np

from astropy.io import fits
from astropy.table import Table, Column, join
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy import wcs

import os
import subprocess
import glob
import logging
from scipy.ndimage import gaussian_filter
from scipy.interpolate import griddata

# personal imports
from .Filaments import Filaments

class Filaments2Files(Filaments):
    """
    Representation of extracting filaments in x-ray using optical data
    
    The Filament2Files class executes the following:
    1. 

    Notes
    -----
    Required softwares: eSASS

    Examples
    --------

    """
    def __init__(self, this_filament_obj: object = Filaments, eRASS1_dir: str = None,\
    data_location: str = None):
        """
        Initialized when an object of the class Filaments2Files is created
        """
        logging.basicConfig(level = logging.INFO)
        self.logger = logging.getLogger(type(self).__name__)
        
        if eRASS1_dir is None:
            self.eRASS1_dir = '/data53s/mpecl_erass1/liuang/erass1_test'
            self.eRASS1_data_dir = '/data53s/mpecl_erass1/data_erass1'
        if data_location is None:
            data_location="/data53s/shreeram/Filament_stacking/data"

        if this_filament_obj is None:
            raise Warning("`this_filament_obj` not set. It must belong to class `Filaments`.")
        
        self.this_filament_obj = this_filament_obj
        self.__repr__()

    def __repr__(self) -> str:
        """
        '__repr__' magic method prints the content of the object
        """
        return f"<Filaments2Files(this_filament_obj={self.this_filament_obj.filament_catalogue_name})>"

    def get_filament_lengths(self, fil_length_cutoff: float = 8, width: float = 2):
        """Method for obtaining an estimate of filament lengths, which are generated from the following algorithms:
        - SCMC
        - TTP
        """
        self.fil_length_cutoff = fil_length_cutoff

        # get the filament points
        if self.this_filament_obj.filament_catalogue_name == 'Tempel_2014_SDSS':
            fil_table, ra_fil, dec_fil = self.this_filament_obj.Tempel_2014_SDSS()
            overlap = self.this_filament_obj.get_overlapping_filaments(ra_fil=ra_fil, dec_fil=dec_fil)
            if len(overlap) > 0:
                # get the overlapping region
                fil_point_IDs = np.unique(fil_table[1][:, 0][overlap.astype(int)])
                choose_overlap = np.in1d(fil_table[0][:, 0], fil_point_IDs)
            
                # build constrain on filament lengths
                fil_lengths = fil_table[0][:, 2][choose_overlap]
                
                ll = np.array(fil_lengths) >= (self.fil_length_cutoff-width)
                ul = np.array(fil_lengths) <= (self.fil_length_cutoff+width)
                choose_lengths = np.array(ll) + np.array(ul)
                fil_IDs = fil_table[0][:, 0][choose_overlap][choose_lengths]
                

                ra_dec_fil_points = []
                # get coordinates belonging to the same filament
                self.logger.info(f" {len(fil_IDs)} filaments with lengths {fil_length_cutoff}+/-{width} Mpc")
                for fil_ID in fil_IDs:
                    choose_coordinates = np.where(fil_table[1][:, 0] == fil_ID)
                    fil_point_properties = fil_table[1][choose_coordinates]
                    
                    x, y, z = fil_point_properties[:, 4], fil_point_properties[:, 5], fil_point_properties[:, 6]
                    c = SkyCoord(x=x, y=y, z=z, unit='Mpc', representation_type='cartesian')

                    ra_single_fil_points, dec_single_fil_points = c.fk5.ra, c.fk5.dec
                    
                    ra_dec_fil_points.append([ra_single_fil_points, dec_single_fil_points])

                self.logger.info(f" Size of filament points array {np.shape(ra_dec_fil_points)}")
            return np.array([ra_dec_fil_points], dtype=object)
        else:
            return 0