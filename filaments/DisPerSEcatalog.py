"""
DisPerSEcatalog.py

This python file owns the class for handling the catalog outputted by the DisPerSE algorithm

Script written by: Soumya Shreeram
Date created: 17th August 2022
Contact: shreeram@mpe.mpg.de
"""
import numpy as np

from astropy.io import fits
from astropy.table import QTable, Table, Column, join
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy import wcs
from astropy.cosmology import z_at_value, Planck18
from gdpyc import GasMap

from astrotools import healpytools as hpt

import os
import subprocess
import glob
import logging
from scipy.ndimage import gaussian_filter
from scipy.interpolate import griddata

import healpy as hp
from .healpyRoutines import produce_healpy_map, healpy_mollview, ra_dec_2_theta_phi

class DisPerSEcatalog:
    """
    Class to represent the catalogs outputted by the DisPerSE filament finding algorithms
    """
    def __init__(self, base_dir: str = None, data_set: str = 'legacy_north_dis', smoothing_density_f: str = "SD1", persistence: float = 3, smoothing_skeleton: str = "S001", section_keyword: str = 'cp', ra_c: float = None, dec_c: float = None):
        """
        Parameters
        ----------
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
        logging.basicConfig(level = logging.INFO)
        self.logger = logging.getLogger(type(self).__name__)
        
        if base_dir is None:
            base_dir = "/data53s/shreeram/Filament_stacking"

        self.code_dir = f"{base_dir}/code"
        self.image_dir = f"{base_dir}/images"
        self.data_dir = f"{base_dir}/data"
        self.catalog_dir = f"{self.data_dir}/Filament_catalogues/Malavasi_2020_SDSS"
        self.catalog_dir_reformatted = f"{self.catalog_dir}/Reformatted"

        # decides filename
        if smoothing_density_f != 'None' and smoothing_skeleton != 'None':
            filename = f"{data_set}.dat.NDnet.{smoothing_density_f}.NDnet_s{persistence}.up.NDskl.BRK.{smoothing_skeleton}.a.NDskl"
        elif smoothing_density_f == 'None' and smoothing_skeleton != 'None':
            filename = f"{data_set}.dat.NDnet_s{persistence}.up.NDskl.BRK.{smoothing_skeleton}.a.NDskl"
        elif smoothing_skeleton == 'None' and smoothing_density_f != 'None':
            filename = f"{data_set}.dat.NDnet.{smoothing_density_f}.NDnet_s{persistence}.up.NDskl.BRK.a.NDskl"
        else:
            filename = f"{data_set}.dat.NDnet_s{persistence}.up.NDskl.BRK.a.NDskl"
        # dedice max redshift
        if data_set == 'legacy_north_dis':
            zmax = 0.5
        if data_set == 'lc_north_dis':
            zmax = 1.0

        if ra_c is None and dec_c is None:
            if data_set == 'legacy_north_dis':
                ra_c, dec_c = 186.183, 26.845
            if data_set == 'lc_north_dis':
                ra_c, dec_c = 184.894, 28.153

        self.base_dir = base_dir

        self.data_set = data_set
        self.zmax = zmax
        self.smoothing_density_f = smoothing_density_f
        self.persistence = persistence
        self.smoothing_skeleton = smoothing_skeleton
        self.catalog_filename = f"{self.catalog_dir}/{filename}"
        self.section_keyword = section_keyword
        self.ra_c = ra_c
        self.dec_c = dec_c

        self.base_name = f"{self.catalog_dir_reformatted}/{self.data_set}_{self.smoothing_density_f}_{self.persistence:.1f}_{self.smoothing_skeleton}"
        self.reformatted_filename = f"{self.base_name}_{self.section_keyword}"
        self.reformatted_extras_filename = f"{self.reformatted_filename}_extras"
        self.logger.info(self.__repr__())
        self.divide_file_sections()

    def __repr__(self) -> str:
        """
        '__repr__' magic method prints the content of the object
        """
        return f"<DisPerSEcatalog(data_set={self.data_set}, smoothing_density_f={self.smoothing_density_f}, persistence={self.persistence}, smoothing_skeleton={self.smoothing_skeleton})>"

    def divide_file_sections(self, filename = None):
        """Method to divide the catalog ascii file into sections 

        Parameters
        ----------
        filename :: str
            The file to open and divide into sections

        Returns
        -------
        cp_start_at, fil_start_at, cp_data_start_at :: int
            the line number on the file where the "critical points", "filaments" and the "critical points data" section start at

        ncrit_line_no, nfil_line_no, nf_line_no :: int
            the line giving information about the number of "critical points", "filaments" and the "critical points data"
        """
        if filename is None:
            filename = self.catalog_filename

        f = open(filename, 'r')
        for l, line in enumerate(f):
            line = line.strip()
            
            # Marks the beginning of the critical points (CP) section
            if line == '[CRITICAL POINTS]':
                cp_start_at = l
                # The number of critical points (CP)
                ncrit_line_no = l + 1
                
            # Marks the beginning of the filaments section
            if line == '[FILAMENTS]':
                fil_start_at = l
                # Total number of filaments
                nfil_line_no = l + 1
                
            # Marks the beginning of the CP data section
            if line == '[CRITICAL POINTS DATA]':
                cp_data_start_at = l
                # Number of fields associated to each CP.
                nf_line_no = l + 1
        f.close()

        self.cp_start_at = cp_start_at
        self.ncrit_line_no = ncrit_line_no

        self.fil_start_at = fil_start_at
        self.nfil_line_no = nfil_line_no

        self.cp_data_start_at = cp_data_start_at
        self.nf_line_no = nf_line_no

    def get_properties(self):
        """
        Method to get the number of "critical points", "filaments" and the "critical points data"
        """
        f = open(self.catalog_filename, 'r')
        for l, line in enumerate(f):
            line = line.strip()

            if l == self.ncrit_line_no:
                ncrit = int(line)
                self.logger.info(f"ncrit={ncrit}")
            if l == self.nfil_line_no:
                nfil = int(line)

            if l == self.nf_line_no:
                nf = int(line)
        f.close()
        
        self.ncrit = ncrit
        self.nfil = nfil
        self.nf = nf
        self.logger.info(f"ncrit={ncrit}, nfil={nfil}, nf={nf}")  

    def read_catalog_file(self):
        """
        Method to read the ascii file that is chosen with the input parameters of the object
        """
        f = open(self.catalog_filename, 'r')
           
        # [CRITICAL POINTS] 
        # -----------------
        if self.section_keyword == 'cp':
            output_ncrit_list, list_entry_lengths, set_group = [], [], []
            for l, line in enumerate(f):
                if (l > self.ncrit_line_no) and l < (self.nfil_line_no-1):
                    line = line.strip()
                    line = np.array(line.split()).astype(float)
                    list_entry_lengths.append(len(line))
                    output_ncrit_list.append(line)
            f.close()
            output_ncrit_list = np.array(output_ncrit_list, dtype=object)

            # using an index to mark each code block within the ascii file
            counter = 0
            for idx, val in enumerate(list_entry_lengths):
                if val == 7:
                    counter += 1
                set_group.append(counter) 
            return output_ncrit_list, set_group

        # [FILAMENTS] 
        # -----------------
        if self.section_keyword == 'fil':
            output_nfil_list, set_group = [], []
            for l, line in enumerate(f):
                line = line.strip()
                if (l > self.nfil_line_no) and l < (self.nf_line_no-1):
                    line = np.array(line.split()).astype(float)
                    output_nfil_list.append(line)
            f.close()
            output_nfil_list = np.array(output_nfil_list, dtype=object)

            # using an index to mark each code block within the ascii file
            counter = 0
            for idx, val in enumerate(output_nfil_list):
                if idx == 0:
                    n_sample_pts = int(val[-1])
                    next_break = int(idx + n_sample_pts + 1)
                    counter += 1
                    
                if idx == int(next_break):
                    n_sample_pts = val[-1]
                    next_break = idx + n_sample_pts + 1
                    
                    # update the counter
                    counter += 1
                    
                set_group.append(counter) 
            return output_nfil_list, set_group

    def convert_ascii2fits(self, clobber: bool = False):
        """
        Method to convert ascii to fits based on the section that is prioritized 
        """
        self.clobber = clobber
        if self.section_keyword == 'cp':
            self.convert_ascii2fits_cp()

        if self.section_keyword == 'fil':
            self.convert_ascii2fits_fil()

    def get_ra_dec_z(self, clobber: bool = False):
        """
        Method to get the ra, dec, and redshift of the section that is concerned 
        """
        self.clobber = clobber
        if self.section_keyword == 'cp':
            self.get_ra_dec_z_cp()

        if self.section_keyword == 'fil':
            self.get_ra_dec_z_fil()

    def convert_ascii2fits_cp(self):
        """
        Method to convert the ascii file for the critical points section to fits format
        """
        if not os.path.exists(self.catalog_filename) or self.clobber:
            output_ncrit_list, set_group = self.read_catalog_file()

            grouped_idx_arr = np.array([], dtype=object)

            cp_type_arr, nfil_cp_arr = np.array([]), np.array([])
            pos_x_arr, pos_y_arr, pos_z_arr = np.array([]), np.array([]), np.array([])
            cp_idx_end_2Darr, fil_idx_2Darr = [], []

            for i in range(1, np.max(set_group)+1):
                # indicies belonging to the same block
                grouped_idx = np.where(np.array(set_group) == ( i ))[0]
                
                # variables in that specific block
                cp_type_arr = np.append(cp_type_arr, int(output_ncrit_list[grouped_idx[0]][0]))
                pos_x_arr = np.append(pos_x_arr, output_ncrit_list[grouped_idx[0]][1])
                pos_y_arr = np.append(pos_y_arr, output_ncrit_list[grouped_idx[0]][2])
                pos_z_arr = np.append(pos_z_arr, output_ncrit_list[grouped_idx[0]][3])
                
                # the number of filaments connected to the critical point 
                nfil_cp = int(output_ncrit_list[grouped_idx[1]][0])
                nfil_cp_arr = np.append(nfil_cp_arr, nfil_cp)
                cp_idx_end_arr, fil_idx_arr = [], []
                for n in range(nfil_cp):
                    number = int(2+n)
                    
                    cp_idx_end, fil_idx = output_ncrit_list[grouped_idx[number]]
                    cp_idx_end_arr.append(int(cp_idx_end)) 
                    fil_idx_arr.append(int(fil_idx))
                
                cp_idx_end_2Darr.append(', '.join( map(str, cp_idx_end_arr)) )
                fil_idx_2Darr.append( ', '.join( map(str, fil_idx_arr) ) )

            table = QTable(data=[cp_type_arr, 
                pos_x_arr,
                pos_y_arr,
                pos_z_arr,
                nfil_cp_arr,
                cp_idx_end_2Darr,
                fil_idx_2Darr
               ], 
               names = [ 'cp_type', 'pos_x', 'pos_y', 'pos_z', 'nfil_cp', 'cp_idx_end_2Darr', 'fil_idx_2Darr'],
               dtype = ['int64', 'float64', 'float64', 'float64', 'int64', 'str', 'str']
               )
            self.logger.info(" Generated table")
            
            table.write(f'{self.reformatted_filename}.fit', overwrite=True)
            self.logger.info(f" Saved table")
        else:
            self.logger.info(" File already exists")

    def convert_ascii2fits_fil(self):
        """
        Method to convert the ascii file for the filaments section to fits format
        """
        if not os.path.exists(self.catalog_filename) or self.clobber:
            output_nfil_list, set_group = self.read_catalog_file()
            
            grouped_idx_arr = np.array([], dtype=object)

            cp1_arr, cp2_arr, n_samp_arr = np.array([]), np.array([]), np.array([])
            pos_x_arr, pos_y_arr, pos_z_arr = np.array([]), np.array([]), np.array([])
            
            for i in range(1, np.max(set_group)+1):
                # indicies belonging to the same block
                grouped_idx = np.where(np.array(set_group) == ( i ))[0]
                
                # variables in that specific block
                cp1_arr = np.append(cp1_arr, int(output_nfil_list[grouped_idx[0]][0]))
                cp2_arr = np.append(cp2_arr, int(output_nfil_list[grouped_idx[0]][1]))

                n_samp = int(output_nfil_list[grouped_idx[0]][2])
                n_samp_arr = np.append(n_samp_arr, n_samp)
                
                
                # the number of filaments connected to the critical point 
                pos_x_temp, pos_y_temp, pos_z_temp = [], [], []
                for n in range(1, n_samp):
                    number = int(n)
                    
                    pos_x, pos_y, pos_z = output_nfil_list[grouped_idx[n]]
                    pos_x_temp.append(pos_x)
                    pos_y_temp.append(pos_y)
                    pos_z_temp.append(pos_z)
                             
                pos_x_arr = np.append(pos_x_arr,  ', '.join(map(str, pos_x_temp)) )
                pos_y_arr = np.append(pos_y_arr,  ', '.join(map(str, pos_y_temp)) )
                pos_z_arr = np.append(pos_z_arr,  ', '.join(map(str, pos_z_temp)) )
                
            table = QTable(data=[cp1_arr, 
                cp2_arr,
                n_samp_arr,
                pos_x_arr,
                pos_y_arr,
                pos_z_arr
               ], 
               names = [ 'cp1', 'cp2', 'n_samp', 'pos_x_arr', 'pos_y_arr', 'pos_z_arr'],
               dtype = ['int64', 'int64', 'int64', 'str', 'str', 'str']
               )
            self.logger.info(" Generated table")
            
            table.write(f'{self.reformatted_filename}.fit', overwrite=True)
            self.logger.info(" Saved table")
        else:
            self.logger.info(" File already exists")

    def get_ra_dec_z_fil(self, zmin: float = 0):
        """Method to get the ra, dec and redshift from the positions in cartesian format (x, y, z)
        Outputs
        -------
        ra_arr, dec_arr :: numppy ndarray
            the arr holds the ra, dec of each filament point outputted in the catalog
        tracker_arr :: numpy ndarray
            this arr tracks the "block" that the (x, y, z) coordinates belong to
        Z_arr :: numpy ndarray
            this array hold the redshifts for every ra, dec, point generated

        """
        t = Table.read(f'{self.reformatted_filename}.fit')
        t_cp = Table.read(f'{self.base_name}_cp.fit')

        cp1, cp2 = t['cp1'], t['cp2']
        counter = 0
        ra_arr, dec_arr, dist_arr, tracker_arr = np.array([]), np.array([]), np.array([]), np.array([])
        for i in range(len(t)):
            counter += 1
            pos_x_cp1, pos_y_cp1, pos_z_cp1 = t_cp[cp1[i]]['pos_x'],  t_cp[cp1[i]]['pos_y'],  t_cp[cp1[i]]['pos_z']
            pos_x = np.array(t[i]['pos_x_arr'].split(',')).astype(float)
            pos_y = np.array(t[i]['pos_y_arr'].split(',')).astype(float)
            pos_z = np.array(t[i]['pos_z_arr'].split(',')).astype(float)
            pos_x_cp2, pos_y_cp2, pos_z_cp2 = t_cp[cp2[i]]['pos_x'],  t_cp[cp2[i]]['pos_y'],  t_cp[cp2[i]]['pos_z']

            pos_x = np.insert(pos_x, [0, -1], [pos_x_cp1, pos_x_cp2])
            pos_y = np.insert(pos_y, [0, -1], [pos_y_cp1, pos_y_cp2])
            pos_z = np.insert(pos_z, [0, -1], [pos_z_cp1, pos_z_cp2])
    
            c = SkyCoord(x=pos_x, y=pos_y, z=pos_z, unit='Mpc', representation_type='cartesian')    
            ra, dec, dist = c.fk5.ra, c.fk5.dec, c.fk5.distance
            
            ra_arr  = np.append( ra_arr, ra.value)
            dec_arr  = np.append( dec_arr, dec.value)
            dist_arr = np.append( dist_arr, dist.value)
            tracker_arr = np.append(tracker_arr, np.ones(len(ra))*counter)
        self.logger.info(f" RA and Dec obtained! Table has {len(ra_arr)} rows")
        
        # convert distance to redshift
        Z_arr = np.array([])
        
        for dist in dist_arr:
            z = z_at_value(Planck18.comoving_distance, dist*u.Mpc, zmin= zmin, zmax= self.zmax)
            Z_arr = np.append(Z_arr, z)
        
        self.logger.info(f" Redshifts obtained! Table has {len(Z_arr)} rows")
        table = QTable(data=[tracker_arr,
            ra_arr, 
            dec_arr,
            dist_arr,
            Z_arr], 
           names = [ 'index', 'RA', 'DEC', 'Distance_Mpc', 'redshift'],
           dtype = ['int64', 'float64', 'float64', 'float64', 'float64']
           )
        self.logger.info(" Generated table")
        table.write(f'{self.reformatted_extras_filename}.fit', overwrite=True)
        self.logger.info(" Saved table")

    def get_ra_dec_z_cp(self, zmin: float = 0):
        """Method to get the ra, dec and redshift from the positions in cartesian format (x, y, z) for the cp section
        Outputs
        -------
        ra_arr, dec_arr :: numppy ndarray
            the arr holds the ra, dec of each filament point outputted in the catalog
        tracker_arr :: numpy ndarray
            this arr tracks the "block" that the (x, y, z) coordinates belong to
        Z_arr :: numpy ndarray
            this array hold the redshifts for every ra, dec, point generated

        """
        # read the file 
        t = Table.read(f'{self.reformatted_filename}.fit')
        
        tracker_arr = np.array([])
        
        pos_x, pos_y, pos_z = t['pos_x'], t['pos_y'], t['pos_z']
        c = SkyCoord(x=pos_x, y=pos_y, z=pos_z, unit='Mpc', representation_type='cartesian')    
        ra_arr, dec_arr, dist_arr = c.fk5.ra, c.fk5.dec, c.fk5.distance
        self.logger.info(f" RA and Dec obtained! Table has {len(ra_arr)} rows.\n Distance has units: {dist_arr.unit}")
        
        # convert distance to redshift
        Z_arr = np.array([])
        
        for i, dist in enumerate(dist_arr):
            z = z_at_value(Planck18.comoving_distance, dist, zmin= zmin, zmax= self.zmax)
            Z_arr = np.append(Z_arr, z)

            tracker_arr = np.append(tracker_arr, i)
        
        self.logger.info(f" Redshifts obtained! Table has {len(Z_arr)} rows")
        table = QTable(data=[tracker_arr,
            ra_arr, 
            dec_arr,
            dist_arr,
            Z_arr], 
           names = [ 'index', 'RA', 'DEC', 'Distance_Mpc', 'redshift'],
           dtype = ['int64', 'float64', 'float64', 'float64', 'float64']
           )
        self.logger.info(" Generated table")
        
        table.write(f'{self.reformatted_extras_filename}.fit', overwrite=True)
        self.logger.info(" Saved table")

    def get_filament_lengths(self, NSIDE=1024):
        """
        Function to get the individual filament lengths
        """
        # read the file 
        t = Table.read(f'{self.reformatted_filename}.fit')
        t_cp = Table.read(f'{self.base_name}_cp.fit')

        cp1, cp2 = t['cp1'], t['cp2']
        length_fil_arr = np.array([])
        for i in range(len(t)):
            pos_x_cp1, pos_y_cp1, pos_z_cp1 = t_cp[cp1[i]]['pos_x'],  t_cp[cp1[i]]['pos_y'],  t_cp[cp1[i]]['pos_z']
            pos_x = np.array(t[i]['pos_x_arr'].split(',')).astype(float)
            pos_y = np.array(t[i]['pos_y_arr'].split(',')).astype(float)
            pos_z = np.array(t[i]['pos_z_arr'].split(',')).astype(float)
            pos_x_cp2, pos_y_cp2, pos_z_cp2 = t_cp[cp2[i]]['pos_x'],  t_cp[cp2[i]]['pos_y'],  t_cp[cp2[i]]['pos_z']

            pos_x = np.insert(pos_x, [0, -1], [pos_x_cp1, pos_x_cp2])
            pos_y = np.insert(pos_y, [0, -1], [pos_y_cp1, pos_y_cp2])
            pos_z = np.insert(pos_z, [0, -1], [pos_z_cp1, pos_z_cp2])

            distances = np.array([])
            for point in range(len(pos_x)-1):
                x1, x2 = pos_x[point], pos_x[point+1]
                y1, y2 = pos_y[point], pos_y[point+1]
                z1, z2 = pos_z[point], pos_z[point+1]
                distance = np.sqrt( (x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2 )
                distances = np.append(distances, distance)
                
            length_fil_arr = np.append(length_fil_arr, np.sum(distances))
        
        # delete column if it exists from previous runs
        if 'Fil_lengths_Mpc' in t.colnames :
            t.remove_column('Fil_lengths_Mpc')
            
        self.logger.info(f"Adding new column with filament lengths, len({len(length_fil_arr)}). Table has length: {len(t)}")
        t.add_column(length_fil_arr, name='Fil_lengths_Mpc')
        t.write(f'{self.reformatted_filename}.fit', overwrite=True)
        self.logger.info(" Modified saved table")

    def select_low_nh_filaments(self, res='high', return_pixel_idx=False, nh_cut = 0.1, nh_key='conservative'):

        """
        Function to get the Nh value for the points where the ra, and dec is given
        """
        # read the file 
        t = Table.read(f'{self.reformatted_filename}.fit')
        t_cp = Table.read(f'{self.base_name}_cp.fit')
        cp1, cp2 = t['cp1'], t['cp2']
        
        nh_flag_arr, nh_vals_arr = np.array([]), np.array([])
        for i in range(len(t)):
            pos_x_cp1, pos_y_cp1, pos_z_cp1 = t_cp[cp1[i]]['pos_x'],  t_cp[cp1[i]]['pos_y'],  t_cp[cp1[i]]['pos_z']
            pos_x = np.array(t[i]['pos_x_arr'].split(',')).astype(float)
            pos_y = np.array(t[i]['pos_y_arr'].split(',')).astype(float)
            pos_z = np.array(t[i]['pos_z_arr'].split(',')).astype(float)
            pos_x_cp2, pos_y_cp2, pos_z_cp2 = t_cp[cp2[i]]['pos_x'],  t_cp[cp2[i]]['pos_y'],  t_cp[cp2[i]]['pos_z']

            pos_x = np.insert(pos_x, [0, -1], [pos_x_cp1, pos_x_cp2])
            pos_y = np.insert(pos_y, [0, -1], [pos_y_cp1, pos_y_cp2])
            pos_z = np.insert(pos_z, [0, -1], [pos_z_cp1, pos_z_cp2])
            
            c = SkyCoord(x=pos_x, y=pos_y, z=pos_z, unit='Mpc', representation_type='cartesian')    
            ra_fil, dec_fil = c.fk5.ra+self.ra_c*u.degree, c.fk5.dec+self.dec_c*u.degree
        
            coords=SkyCoord(ra_fil, dec_fil, unit = 'deg')
            if res=='high':
                nh_vals = GasMap.nh(coords, nhmap = 'HI4PI', hires = True).value/1e22
                NSIDE = 1024
            if res=='low':
                nh_vals = GasMap.nh(coords, nhmap = 'HI4PI', hires = False).value/1e22
                NSIDE = 512
                
            foreground_dominated = np.where(nh_vals>nh_cut)[0]
            if nh_key == 'conservative':
                if len(foreground_dominated) > 0:
                    flag = 0
                else:
                    flag = 1

            if nh_key == 'relaxed':
                if len(foreground_dominated) == len(pos_x):
                    flag = 0
                else:
                    flag = 1
            nh_flag_arr = np.append(nh_flag_arr, flag)
            nh_vals_arr = np.append(nh_vals_arr,  ', '.join(map(str, nh_vals)) )
                
        if f'Nh_flag_{nh_key}' in t.colnames:
            t.remove_column(f'Nh_flag_{nh_key}')
        if f'Nh_val_{nh_key}' in t.colnames:
            t.remove_column(f'Nh_val_{nh_key}')

        self.logger.info(f"Adding new column with Nh_flag_{nh_key}, len({len(nh_vals_arr)}). Table has length: {len(t)}")
        t.add_column(nh_flag_arr, name=f'Nh_flag_{nh_key}')
        t.add_column(nh_vals_arr, name=f'Nh_val_{nh_key}')
        t.write(f'{self.reformatted_filename}.fit', overwrite=True)
        self.logger.info(" Modified saved table")
        self.nh_key = nh_key

    def get_orientation_filaments(self):
        """Function to calculate the orientation of the filaments with respect to the line of sight (x-axis)
        """
        unit_vector_x = [1, 0, 0]

        # read the file 
        t = Table.read(f'{self.reformatted_filename}.fit')
        t_cp = Table.read(f'{self.base_name}_cp.fit')
        cp1, cp2 = t['cp1'], t['cp2']
        
        angle_rad_arr = np.array([])
        for i in range(len(t)):
            pos_x_cp1, pos_y_cp1, pos_z_cp1 = t_cp[cp1[i]]['pos_x'],  t_cp[cp1[i]]['pos_y'],  t_cp[cp1[i]]['pos_z']
            pos_x = np.array(t[i]['pos_x_arr'].split(',')).astype(float)
            pos_y = np.array(t[i]['pos_y_arr'].split(',')).astype(float)
            pos_z = np.array(t[i]['pos_z_arr'].split(',')).astype(float)
            pos_x_cp2, pos_y_cp2, pos_z_cp2 = t_cp[cp2[i]]['pos_x'],  t_cp[cp2[i]]['pos_y'],  t_cp[cp2[i]]['pos_z']

            pos_x = np.insert(pos_x, [0, -1], [pos_x_cp1, pos_x_cp2])
            pos_y = np.insert(pos_y, [0, -1], [pos_y_cp1, pos_y_cp2])
            pos_z = np.insert(pos_z, [0, -1], [pos_z_cp1, pos_z_cp2])

            average_fil_vector = [np.mean(pos_x), np.mean(pos_y), np.mean(pos_z)]
            
            unit_fil_vector = average_fil_vector/np.linalg.norm(average_fil_vector)

            dot_product = np.dot(unit_fil_vector, unit_vector_x)
            angle = np.arccos(dot_product)

            angle_rad_arr = np.append(angle_rad_arr, angle)
        
        # delete column if it exists from previous runs
        if 'Orientation_radian' in t.colnames :
            t.remove_column('Orientation_radian')
            
        self.logger.info(f"Adding new column with filament orientation, len({len(angle_rad_arr)}). Table has length: {len(t)}")
        t.add_column(angle_rad_arr, name='Orientation_radian')
        t.write(f'{self.reformatted_filename}.fit', overwrite=True)
        self.logger.info(" Modified saved table")
        return

    def get_elongation_filaments(self):
        """Function to get the RMS of the filaments
        """
        unit_vector_x = [1, 0, 0]

        # read the file 
        t = Table.read(f'{self.reformatted_filename}.fit')
        t_cp = Table.read(f'{self.base_name}_cp.fit')
        cp1, cp2 = t['cp1'], t['cp2']
        
        elongation_arr = np.array([])
        for i in range(len(t)):
            pos_x_cp1, pos_y_cp1, pos_z_cp1 = t_cp[cp1[i]]['pos_x'],  t_cp[cp1[i]]['pos_y'],  t_cp[cp1[i]]['pos_z']
            pos_x = np.array(t[i]['pos_x_arr'].split(',')).astype(float)
            pos_y = np.array(t[i]['pos_y_arr'].split(',')).astype(float)
            pos_z = np.array(t[i]['pos_z_arr'].split(',')).astype(float)
            pos_x_cp2, pos_y_cp2, pos_z_cp2 = t_cp[cp2[i]]['pos_x'],  t_cp[cp2[i]]['pos_y'],  t_cp[cp2[i]]['pos_z']

            pos_x = np.insert(pos_x, [0, -1], [pos_x_cp1, pos_x_cp2])
            pos_y = np.insert(pos_y, [0, -1], [pos_y_cp1, pos_y_cp2])
            pos_z = np.insert(pos_z, [0, -1], [pos_z_cp1, pos_z_cp2])

            fil_vector = np.array([[x, y, z] for x, y, z in zip(pos_x, pos_y, pos_z)])
            straight_vector = fil_vector[-1]-fil_vector[0]
            straight_vector_norm = np.sqrt(np.sum(straight_vector**2))

            elongation = straight_vector_norm/t[i]['Fil_lengths_Mpc']
            elongation_arr = np.append(elongation_arr, elongation)

        # delete column if it exists from previous runs
        if 'Elongation' in t.colnames :
            t.remove_column('Elongation')
            
        self.logger.info(f"Adding new column with filament Elongation, len({len(elongation_arr)}). Table has length: {len(t)}")
        t.add_column(elongation_arr, name='Elongation')
        t.write(f'{self.reformatted_filename}.fit', overwrite=True)
        self.logger.info(" Modified saved table")
        return

    def check_if_filament_in_eRASSde(self):
        """Function to check if the filaments are in the German-half of the eRASS sky
        """
        # read the file 
        c_center = SkyCoord(ra=self.ra_c, dec=self.dec_c, frame='icrs', unit=u.deg)
        l_cen, b_cen = c_center.galactic.l, c_center.galactic.b 
        self.l_cen, self.b_cen = l_cen, b_cen
    
        t = Table.read(f'{self.reformatted_filename}.fit')
        t_cp = Table.read(f'{self.base_name}_cp.fit')
        cp1, cp2 = t['cp1'], t['cp2']
        
        ownership_arr = np.array([])
        for i in range(len(t)):
            pos_x_cp1, pos_y_cp1, pos_z_cp1 = t_cp[cp1[i]]['pos_x'],  t_cp[cp1[i]]['pos_y'],  t_cp[cp1[i]]['pos_z']
            pos_x = np.array(t[i]['pos_x_arr'].split(',')).astype(float)
            pos_y = np.array(t[i]['pos_y_arr'].split(',')).astype(float)
            pos_z = np.array(t[i]['pos_z_arr'].split(',')).astype(float)
            pos_x_cp2, pos_y_cp2, pos_z_cp2 = t_cp[cp2[i]]['pos_x'],  t_cp[cp2[i]]['pos_y'],  t_cp[cp2[i]]['pos_z']

            # add the critical points to the filaments
            pos_x = np.insert(pos_x, [0, -1], [pos_x_cp1, pos_x_cp2])
            pos_y = np.insert(pos_y, [0, -1], [pos_y_cp1, pos_y_cp2])
            pos_z = np.insert(pos_z, [0, -1], [pos_z_cp1, pos_z_cp2])
            
            c = SkyCoord(x=pos_x, y=pos_y, z=pos_z, unit='Mpc', representation_type='cartesian') 
            # shift the center using the reference coordinates
            l, b = c.galactic.l + l_cen, c.galactic.b + b_cen

            # decide which half of the sky the filaments lie in
            if np.all((l.value >= 180) & (l.value <= 360)):
                ownership_arr = np.append(ownership_arr, 'DE')
            else:
                ownership_arr = np.append(ownership_arr, 'RU')

        # delete column if it exists from previous runs
        if 'Ownership' in t.colnames:
            t.remove_column('Ownership')
            
        self.logger.info(f"Adding new column with filament Ownership, len({len(ownership_arr)}). Table has length: {len(t)}")
        t.add_column(ownership_arr, name='Ownership')
        t.write(f'{self.reformatted_filename}.fit', overwrite=True)
        self.logger.info(" Modified saved table")
        return

    def count_filaments(self, len_low=30, len_high=100, ang_low=40, ang_high=90):
        """This function gives the number of available filaments after all the cuts have been applied
        
        Parameters
        ----------
        len_low, len_high:: floats
            The lower and upper limit of the selected filaments
        ang_low, ang_high :: floats
            The lower and upper limits of the orientation cuts

        Returns
        -------
        number of filaments :: int
            The total number of filaments avaiable after ownership, length and orientation cuts
        select_lengths :: ndarray of bool
            The boolean array used for selecting lengths
        orientation_cut :: ndarray of bool
            The boolean array used for selecting orientations in the given range
        """
        table_fil = Table.read(f'{self.reformatted_filename}.fit')
        length_fil_arr = table_fil['Fil_lengths_Mpc']
        select_lengths = (length_fil_arr >= len_low) & (length_fil_arr <= len_high)
    
        angle = np.rad2deg(table_fil['Orientation_radian'])
        orientation_cut = (angle >= ang_low) & (angle <= ang_high)
    
        number_selected_filaments = len(table_fil[(table_fil['Ownership']=='DE') & orientation_cut & select_lengths])

        self.logger.info(f"total filaments available {len(table_fil)}")
        self.logger.info("===========\n")

        self.logger.info("German half of the sky has:")
        self.logger.info(f"Total {len(table_fil[(table_fil['Ownership']=='DE') ])}")
        self.logger.info(f"Length and orientation cut {number_selected_filaments}")
        self.logger.info(f"Only length cut {len(table_fil[(table_fil['Ownership']=='DE') & select_lengths])}")
        self.logger.info(f"Only orientation cut {len(table_fil[(table_fil['Ownership']=='DE') & orientation_cut ])}")
        self.logger.info("===========\n")
        
        self.logger.info("Russian half of the sky has:")
        self.logger.info(f"Total {len(table_fil[(table_fil['Ownership']=='RU')])}")
        self.logger.info(f"Length and orientation cut {len(table_fil[(table_fil['Ownership']=='RU') & select_lengths & orientation_cut])}")
        self.logger.info(f"Only length cut {len(table_fil[(table_fil['Ownership']=='RU') & select_lengths])}")
        self.logger.info(f"Only orientation cut {len(table_fil[(table_fil['Ownership']=='RU') & orientation_cut ])}")
        self.logger.info("===========\n")
        return number_selected_filaments, select_lengths, orientation_cut
