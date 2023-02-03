"""
ClusterCatalogs.py

This python file owns the class to load and manipulate different cluster catalogs

Script written by: Soumya Shreeram
Date created: 18th May 2022
Contact: shreeram@mpe.mpg.de
"""
# astropy modules
import astropy.units as u
import astropy.io.fits as fits

from astropy.table import Table, Column, join
from astropy.coordinates import SkyCoord
from astropy.cosmology import FlatLambdaCDM, z_at_value

import numpy as np

# system imports
import os
import sys
import importlib as ib
import glob
import gzip
import copy
import logging

# plotting imports
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D

import seaborn as sns
import pandas as pd

import healpy as hp
from healpy.newvisufunc import projview, newprojplot
from .healpyRoutines import produce_healpy_map

class ClusterCatalogs:
    def __init__(self, this_cluster_cat_name: str = None, base_dir: str = None,
        redshift_range: tuple = None):
        """
        Initialized when an object of the class ClusterCatalogs is created

        Parameters
        ----------
        this_cluster_cat_name : string
            Decides which cluster catalog to open for using
        base_dir : string
            Local directory where the code, images, and data folders are contained
        redshift_range : tuple
            Defined the lower and upper limit within which the objects are chosen
        """
        logging.basicConfig(level = logging.INFO)
        self.logger = logging.getLogger(type(self).__name__)
        
        if this_cluster_cat_name is None:
            this_cluster_cat_name = 'eRASS1'
            self.logger.info(f"Default cluster catalog set to `eRASS1`")
            self.logger.info(f"Other options include: PSZ, ACT")

        if base_dir is None:
            base_dir = "/data53s/shreeram/Filament_stacking"
            code_dir = f"{base_dir}/code"
            image_dir = f"{base_dir}/images"
            data_dir = f"{base_dir}/data/Cluster_catalogs" 

        if redshift_range is None:
            z_low, z_high = 0.3, 0.32
            
            self.logger.info(f"`redshift_range` not set. Default set to ({z_low}, {z_high}) ")
            redshift_range = (z_low, z_high)
        
        self.this_cluster_cat_name = this_cluster_cat_name
        self.base_dir = base_dir
        self.code_dir, self.image_dir, self.data_dir = code_dir, image_dir, data_dir 
        self.redshift_range = redshift_range
        self.__repr__()

        # outputs the `cluster_cat_pathname`
        self.get_cluster_cat_pathname()
        

    def __repr__(self) -> str:
        """
        '__repr__' magic method prints the content of the object
        """
        return f"<ClusterCatalogs(this_cluster_cat_name={self.this_cluster_cat_name})>"


    def get_cluster_cat_pathname(self):
        """Function gets the name of the cluster catalog.
        Each cluster catalog has different naming conventions for redshift, ra and dec
        This function defines them for each of the loaded catalogs. 

        The 'combined' option contains the following 11 cluster catalogs:
        (This is defined by the `srccat` column.)
        - ACTDR5
        - MCXC
        - PSZ2
        - RXGCC
        - XCSDR1
        - XClass
        - XXL365
        - eFEDS
        - spt2500d
        - spt_ecs
        - sptpol100d
        """

        if self.this_cluster_cat_name  == 'eRASS1':
            cluster_cat_pathname = f"{self.data_dir}/eRASS:1/erass1_cl_v0.1.fits"
            z_key, ra_key, dec_key = 'BEST_Z', 'RA', 'DEC'

        if self.this_cluster_cat_name  == 'ACT':
            cluster_cat_pathname = f"{self.data_dir}/ACT2020/DR5_cluster-catalog_v1.1_forSZDB.fits"
            z_key, ra_key, dec_key = 'redshift', 'RADeg', 'decDeg'

        if self.this_cluster_cat_name  == 'PSZ':
            cluster_cat_pathname = f"{self.data_dir}/PSZ2v1.fits"
            z_key, ra_key, dec_key = 'REDSHIFT', 'RA', 'DEC'

        if self.this_cluster_cat_name == 'combined':
            cluster_cat_pathname = f"{self.data_dir}/ClGmask_v3.fits"
            z_key, ra_key, dec_key = 'z', 'RA', 'DEC'

        self.cluster_cat_pathname = cluster_cat_pathname
        self.z_key, self.ra_key, self.dec_key = z_key, ra_key, dec_key

        
    def get_cluster_cat_data(self):
        """Function chooses only clusters based on the following criteria:
        1. If the clusters fall into the redshift bin defined in the beginning
        2. In case of eRASS:1 clusters, if they are corrected for the splitting problem
        """

        z_low, z_high = self.redshift_range
        cluster_cat = Table.read(self.cluster_cat_pathname, format='fits')
        
        if self.this_cluster_cat_name == 'combined':
            self.logger.info(f"catalogs in this file {np.unique(cluster_cat['srccat'])}")

        # choose redshift bin
        self.logger.info(f"Selecting in the redshift bin {z_low}<z<{z_high}")
        cluster_cat = cluster_cat[cluster_cat[self.z_key] != np.nan]
        choose_z = (cluster_cat[self.z_key] > z_low) & (cluster_cat[self.z_key] < z_high)
        
        # for eRASS1 need to ensure additional cleaning is applied
        if self.this_cluster_cat_name  == 'eRASS1':
            choose_current_good = (cluster_cat['SPLIT_CLEANED'] == True) 
            sample =  choose_current_good & choose_z
        else:
            sample = choose_z
        
        cluster_cat = cluster_cat[sample]
        z, ra, dec = cluster_cat[self.z_key], cluster_cat[self.ra_key], cluster_cat[self.dec_key]
        return z, np.array(ra), np.array(dec), cluster_cat

    def get_healpy_clusters_map(self, NSIDE: int =256, fil_pixel_cluster: int = 1,
        big_radius=.02, tiny_radius=.01, fil_pixel_cluster_disc=2.75):
        """
        Method defines the cluster map in healpy and plots discs around the cluster
        Parameters
        ----------
        NSIDE : int, default set to 256
            The healpix nside parameter, must be a power of 2, less than 2**30
        big_radius: float, default  set to .04 [radians]
            Defines the outer region to choose around the cluster for drawing circles
        tiny_radius: float, default  set to .03 [radians]
            Defines the inner region --""-- (same as above)
        fil_pixel_cluster_disc : float, default 2.75
            Value by which the helapy map with discs is filled
        """
        z_clu, ra_clu, dec_clu, cluster_cat = self.get_cluster_cat_data()
        cluster_map, clu_pix_idx = produce_healpy_map(NSIDE, ra_clu, dec_clu, return_pixel_idx=True,
                                                fil_val=fil_pixel_cluster)

        vec_clu = hp.ang2vec(ra_clu, dec_clu, lonlat=True)
        cluster_map_with_discs = np.copy(cluster_map)
        for vec in vec_clu:
            disc = hp.query_disc(NSIDE, vec, radius=big_radius, nest=True)
            disc_in = hp.query_disc(NSIDE, vec, radius=tiny_radius, nest=True)
            circle = np.setdiff1d(disc, disc_in)

            cluster_map_with_discs[circle] = fil_pixel_cluster_disc
    
        return cluster_map, cluster_map_with_discs, clu_pix_idx