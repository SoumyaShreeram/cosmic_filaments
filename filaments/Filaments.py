# astropy modules
import astropy.units as u
import astropy.io.fits as fits

from astropy.table import Table, Column, join
from astropy.coordinates import SkyCoord
from astropy.cosmology import FlatLambdaCDM, z_at_value
from astropy.coordinates import SkyCoord

import numpy as np

# system imports
import os
import sys
import importlib as ib
import glob
import gzip
import shutil

# plotting imports
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import norm
from scipy import interpolate
import pandas as pd
import logging

from .utils import aitoffProjection
matplotlib.rcParams['agg.path.chunksize'] = 100000



class Filaments:
    def __init__(self, eRASS4_dir: str = None, eRASS1_dir: str = None, data_location: str = None, filament_catalogue_name: str = None,\
        which_redshifts: str = 'lowz') -> None:
        """Initialized the following when class object is created

        Parameters
        ----------
        eRASS1_dir :: str
            directory where the eRASS1 data is stored

        Notes
        -----
        Data directory much contain:
        1. "SKYMAPS.fits"

        The Filament catalogues are downloaded from public domains. 
        The catalogues used in this class can be found at the follwing links:
        - Tempel_2014_SDSS:
            https://cdsarc.cds.unistra.fr/viz-bin/cat?J/MNRAS/438/3465#/browse
        - Duque_2021_SDSS:
            https://www.javiercarron.com/catalogue
        - Eardley_2015_GAMA:
            http://www.gama-survey.org/dr3/schema/table.php?id=84
        - Yen_Chi_2017_SDSS:
            https://sites.google.com/site/yenchicr/catalogue
        """
        logging.basicConfig(level = logging.INFO)
        self.logger = logging.getLogger(type(self).__name__)

        if eRASS1_dir is None:
            eRASS1_dir = '/data53s/mpecl_erass1/liuang/erass1_test'
            eRASS1_data_dir = '/data53s/mpecl_erass1/data_erass1'
        self.eRASS1_dir = eRASS1_dir
        self.eRASS1_data_dir = eRASS1_data_dir
        
        if eRASS4_dir is None:
            eRASS4_dir = "/data53s/mpecl_erass1/data_s4/"
        self.eRASS4_dir = eRASS4_dir

        if data_location is None:
            data_location="/data53s/shreeram/Filament_stacking/data"
        self.data_location = data_location

        if filament_catalogue_name is None:
            self.logger.info(" no choice for 'filament_catalogue_name' was given. Currently available options are as follows:\
                ['Tempel_2014_SDSS', 'Duque_2021_SDSS', 'Eardley_2015_GAMA', 'Yen_Chi_2017_SDSS']")
            self.logger.info(" Default set to 'Duque_2021_SDSS' ")
            filament_catalogue_name = 'Duque_2021_SDSS'
        
        fil_cat_path = f"{self.data_location}/Filament_catalogues/{filament_catalogue_name}"
        self.filament_catalogue_name = filament_catalogue_name
        self.fil_cat_path = fil_cat_path
        self.which_redshifts = which_redshifts
        self.sns_colors = sns.color_palette("tab10")
        self.__repr__()
        
    def __repr__(self) -> str:
        '''
        '__repr__' magic method prints the content of the object
        '''
        return f"<Filament(filament_catalogue_name={self.filament_catalogue_name})>"
            

    def open_fil_cat(self, return_lengths: bool = False):
        """Method opens the filament catalogue based on the user inout
        """
        self.return_lengths = return_lengths

        if self.filament_catalogue_name == 'Duque_2021_SDSS':
            fil_table, ra_fil, dec_fil = self.Duque_2021_SDSS()
            self.plot_title="Duque et al. 2021"
            
        if self.filament_catalogue_name == 'Eardley_2015_GAMA':
            fil_table, ra_fil, dec_fil = self.Eardley_2015_GAMA()
            self.plot_title = "Eardley et al. 2015"
            
        if self.filament_catalogue_name == 'Tempel_2014_SDSS':
            fil_table, ra_fil, dec_fil = self.Tempel_2014_SDSS()
            self.plot_title = "Tempel et al. 2014"
        
        if self.filament_catalogue_name == 'Yen_Chi_2017_SDSS':
            fil_table, ra_fil, dec_fil = self.Yen_Chi_2017_SDSS()
            self.plot_title = "Yen Chi et al. 2017"

        self.catalog_file_description = f"{self.survey_name}: z={self.redshift_ranges[0]:.1f}-{self.redshift_ranges[1]:.1f}"
        return fil_table, ra_fil, dec_fil

    """
    Functions for Duque et al 2021 Filament catalog 
    """

    def Duque_2021_SDSS(self):
        """Method to open the filament catalogue by Duque et a. 2021
        The catalogue is based on SDSS (BOSS, CMASS, and eBOSS) data and for more details refer to the paper/website:
        https://www.javiercarron.com/catalogue
        """
        if self.which_redshifts == 'lowz':
            redshift_ranges = (0.05, 0.45)
            survey_name = "BOSS"
            file_name = 'Block1.csv'
        if self.which_redshifts == 'midz':
            redshift_ranges = (0.45, 0.7)
            file_name = 'Block2.csv'
            survey_name = "BOSS CMASS"
        if self.which_redshifts == 'highz':
            redshift_ranges = (0.6, 2.2)
            file_name = 'Block3.csv'
            survey_name = r"BOSS+eBOSS"
        self.logger.info(f"Default param 'which_redshifts={self.which_redshifts}'  ")

        self.redshift_ranges  = redshift_ranges
        self.survey_name  =  survey_name
        self.file_name = glob.glob(f"{self.fil_cat_path}/*.fits")
        
        self.catalog_file_path = f"{self.fil_cat_path}/{file_name}"

        # open the chosen file (not the array outputted by glob.glob)
        fil_table = pd.read_csv(self.catalog_file_path)
        
        # convert ra and dec of the filaments into radian + aitoff proj
        ra_fil = np.array(fil_table['RA'])*u.deg
        dec_fil = np.array(fil_table['dec'])*u.deg
        return fil_table, ra_fil, dec_fil
        
    def choose_z_unc(self, fil_table = None, ra_og = None, dec_og = None, 
        unc_percentile_high: float = 40, z_low: float = 0.3, 
        z_high: float = 0.32, cut_in_uncertainty=True):
        """
        This function is applicable to the Duque et al. catalogue only
        
        unc_percentile_high :: int
            The upper limit of the percentile in the uncertainty dist of filaments below which the filament-points are chosen
        z_high :: float
            The upper limit of the redshift slice that is chosen
        z_low :: float
            The lower limit of the redshift slice
        """
        if fil_table is None:
            fil_table, ra_og, dec_og = self.Duque_2021_SDSS()

        chosen_redshift = (fil_table['z_low'] > z_low) & (fil_table['z_low'] < z_high)
        
        if cut_in_uncertainty:
            cut_off_x = np.percentile(fil_table['unc'], unc_percentile_high)
            chosen_uncertainty = fil_table['unc'] < cut_off_x
            sample = chosen_redshift & chosen_uncertainty
        else:
            sample = chosen_redshift
        ra, dec = ra_og[sample], dec_og[sample]
        return ra, dec

    """
    Functions for Eardley_2015_GAMA Filament catalog 
    """

    def Eardley_2015_GAMA(self):
        """Method to open the filament catalogue by Eardley et al. 2015
        The catalogue is based on GAMA data and for more details refer to the article:
        https://ui.adsabs.harvard.edu/abs/2015MNRAS.448.3665E/abstract
        """
        self.survey_name = 'GAMA'
        self.redshift_ranges = (0.04, 0.26)

        self.file_name = glob.glob(f"{self.fil_cat_path}/*.fits")
        self.logger.info(f" Files in the filament catalogue: \n {self.file_name}")
        
        if len(self.file_name) == 1:
            file_name = self.file_name[0]
            self.catalog_file_path = f"{self.fil_cat_path}/{file_name}"
        else:
            self.logger.info(" found more than 1 file in the directory. Something is strange.")

        fil_table = Table.read(file_name, format="fits")
        
        # GeoS4(10): Geometric environment classification of cell computed with 4(10) Mpc/h smoothing
        fil_table = fil_table[fil_table['GeoS4']==2]
        ra_fil = fil_table['RA']
        dec_fil = fil_table['DEC']
        return fil_table, ra_fil, dec_fil 
        
    def Tempel_2014_SDSS(self):
        """Method to open the filament catalogue by Tempel et al. 2014 
        The catalogue is based on SDSS data. For more details refer to the article: 10.1093/mnras/stt2454
        """
        self.survey_name = 'SDSS'
        self.redshift_ranges = (0.05, 0.155)
        self.file_name = glob.glob(f"{self.fil_cat_path}/*.txt")
        self.logger.info(f" Files in the filament catalogue: \n {self.file_name}")

        # if interested in styding filament properties
        self.fil_properties_name = f"{self.fil_cat_path}/fil_properties.txt"
        self.fil_point_properties_name = f"{self.fil_cat_path}/fil_point_properties.txt"
        self.catalog_file_path = f"{self.fil_cat_path}/{self.fil_properties_name}"
        
        # load the data
        fil_properties = np.loadtxt(self.fil_properties_name)
        
        fil_point_properties = np.loadtxt(self.fil_point_properties_name)
        x, y, z = fil_point_properties[:, 4], fil_point_properties[:, 5], fil_point_properties[:, 6]
        c = SkyCoord(x=x, y=y, z=z, unit='Mpc', representation_type='cartesian')

        ra_fil, dec_fil = c.fk5.ra, c.fk5.dec

        fil_table = np.array([fil_properties, fil_point_properties], dtype=object)
        fil_lengths = fil_properties[:, 2]

        if self.return_lengths:
            return fil_table, ra_fil, dec_fil, fil_lengths
        else:
            return fil_table, ra_fil, dec_fil

    def Yen_Chi_2017_SDSS(self):
        """Method to open the filament catalogue by Yen Chi et al. 2017 
        The catalogue is based on SDSS data. For more details refer to these articles: 
        -  arXiv: 1501.05303
        -  arXiv: 1509.06443
        """
        self.survey_name = 'SDSS'
        self.redshift_ranges = (0.05, 0.7)

        self.file_name = glob.glob(f"{self.fil_cat_path}/*.txt")
        self.logger.info(f" Files in the filament catalogue: \n {self.file_name}")

        # the file of interest in particular is 
        self.fil_properties_name = f"{self.fil_cat_path}/dr12_FMaps_full.txt"
        fil_table = np.loadtxt(self.fil_properties_name, skiprows=1)
        ra_fil, dec_fil = fil_table[:, 0]*u.deg, fil_table[:, 1]*u.deg
        return fil_table, ra_fil, dec_fil
        
    def readeRASSskyTileFile(self):
        """Method reads the skytiles and plots them in mollwide projection
        """
        filename = os.path.join(self.eRASS1_dir, "SKYMAPS.fits")
        skyf = Table.read(filename, format='fits')

        owner_de = skyf['OWNER'] & 1 != 0
        sky_ra_min = skyf['RA_MIN'][owner_de]
        sky_ra_max = skyf['RA_MAX'][owner_de]
        sky_dec_min = skyf['DE_MIN'][owner_de]
        sky_dec_max = skyf['DE_MAX'][owner_de]


        eRASS_ra_min, eRASS_dec_min = aitoffProjection(sky_ra_min, sky_dec_min)
        eRASS_ra_max, eRASS_dec_max = aitoffProjection(sky_ra_max, sky_dec_max)
        return eRASS_ra_min, eRASS_dec_min, eRASS_ra_max, eRASS_dec_max
    
    def get_overlapping_filaments(self, ra_fil = None, dec_fil = None):
        """
        Function to check the number filament points that fall into the eRASS sky tiles
        @ra_fil :: array with the ra of the fillament 
        """
        filament_idx_all_sky_tiles = np.array([])
        eRASS_ra_min, eRASS_dec_min, eRASS_ra_max, eRASS_dec_max = self.readeRASSskyTileFile()
        if ra_fil and dec_fil is None:
            _, ra_fil, dec_fil = self.open_fil_cat() 
        ra_fil, dec_fil = aitoffProjection(ra_fil, dec_fil)

        for ra_min, ra_max, dec_min, dec_max in zip(eRASS_ra_min, eRASS_ra_max, eRASS_dec_min, eRASS_dec_max):
            ra_fil_cond = (ra_fil >= ra_min) & (ra_fil <= ra_max)
            dec_fil_cond = (dec_fil >= dec_min) & (dec_fil <= dec_max)
            if_within_sky_tile = np.where(ra_fil_cond & dec_fil_cond)
            
            # save the indicies of the filament points within the sky tile
            if len(if_within_sky_tile)>0:
               filament_idx_all_sky_tiles = np.append(filament_idx_all_sky_tiles, if_within_sky_tile[0])
            else:
                filament_idx_all_sky_tiles = np.append(filament_idx_all_sky_tiles, [None])
        return filament_idx_all_sky_tiles