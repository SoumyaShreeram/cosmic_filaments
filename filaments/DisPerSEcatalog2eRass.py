"""
DisPerSEcatalog2eRass.py

This python file owns the class for generating eRASS products given the DisPerSE catalog outputs

Script written by: Soumya Shreeram
Date created: 29th September 2022
Contact: shreeram@mpe.mpg.de
"""
import numpy as np

from astropy.io import fits
from astropy.table import QTable, Table, Column, join
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy import wcs
from astropy.cosmology import z_at_value, Planck18, FlatLambdaCDM
from gdpyc import GasMap

from astrotools import healpytools as hpt

import os
import subprocess
import glob
import logging
from scipy.ndimage import gaussian_filter
from scipy.interpolate import griddata

import healpy as hp
from .DisPerSEcatalog import DisPerSEcatalog
from .utils import cen_pos, astropy_dist, circle

class DisPerSEcatalog2eRass(DisPerSEcatalog):
    """
    Class to represent and create the file structure for generating images/spectra in X-ray eRosita data given the optical filament catalog by the DisPerSE algorithm by Malavasi et al. (2020). The optical filament catalog is built using the LOWZ+CMASS SDSS surveys

    Data files required:
    --------------------
    1. tm1_2dpsf_190220v03.fits
    2. onaxis_020_RMF_00001.fits
    3. onaxis_020_ARF_00001.fits
    4. psf_prof.dat
    5. SKYMAP_NewOwner.fits
    6. masklist.fits
    """
    def __init__(self, eRASS4_dir: str = None, save_data_dir: str = None, base_dir: str = None, data_set: str = 'lc_north_dis', smoothing_density_f: str = "None", persistence: float = 3, smoothing_skeleton: str = "None", section_keyword: str = 'fil', ra_c: float = None, dec_c: float = None, cosmo: object = None, verbrose: int = 1):
        """
        Initialized when an object of DisPerSEcatalog2eRass is called
        
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
        super().__init__(base_dir, data_set, smoothing_density_f, persistence, smoothing_skeleton, section_keyword, ra_c, dec_c)

        logging.basicConfig(level = logging.INFO)
        self.logger = logging.getLogger(type(self).__name__)

        if eRASS4_dir is None:
            eRASS4_dir = '/data53s/mpecl_erass1/data_s4'
            eRASS1_data_dir = '/data53s/mpecl_erass1/data_erass1'
        self.eRASS4_dir = eRASS4_dir
        self.eRASS1_data_dir = eRASS1_data_dir

        if save_data_dir is None:
            save_data_dir="/he13srv_local/shreeram"
        self.save_data_dir = save_data_dir
        self.data_location = f"{self.save_data_dir}/eRass_products"

        # gets info on the number of filaments 
        self.get_number_filament()

        if cosmo is None:
            cosmo = FlatLambdaCDM(H0=67.77*u.km/u.s/u.Mpc, Om0=0.307115)
        self.cosmo = cosmo

        self.verbrose = verbrose

    def __repr__(self) -> str:
        """
        '__repr__' magic method prints the content of the object
        """
        return f"<DisPerSEcatalog2eRass(data_set={self.data_set}, smoothing_density_f={self.smoothing_density_f}, persistence={self.persistence}, smoothing_skeleton={self.smoothing_skeleton}, section={self.section_keyword})>"

    def get_number_filament(self):
        """Method gets the total number of selected filaments, in addition to the selection arrays for the length and orientation
        """
        number_fils, select_lengths, orientation_cut = super().count_filaments()
        self.number_fils = number_fils
        self.select_lengths = select_lengths
        self.orientation_cut = orientation_cut

    def get_filament_medians(self, zmin: float = 0):
        """Method to get the region around a filament for inputting the values into eSASS tools
        Parameters
        ----------
        @zmin :: float
            the minimun redshift for 
        """
        t = Table.read(f'{self.reformatted_filename}.fit')
        t_cp = Table.read(f'{self.base_name}_cp.fit')

        cp1, cp2 = t['cp1'], t['cp2']
        ra_median_arr, dec_median_arr, dist_median_arr = np.array([]), np.array([]), np.array([])
        
        for i in range(len(t)):
            pos_x_cp1, pos_y_cp1, pos_z_cp1 = t_cp[cp1[i]]['pos_x'],  t_cp[cp1[i]]['pos_y'],  t_cp[cp1[i]]['pos_z']
            pos_x = np.array(t[i]['pos_x_arr'].split(',')).astype(float)
            pos_y = np.array(t[i]['pos_y_arr'].split(',')).astype(float)
            pos_z = np.array(t[i]['pos_z_arr'].split(',')).astype(float)
            pos_x_cp2, pos_y_cp2, pos_z_cp2 = t_cp[cp2[i]]['pos_x'],  t_cp[cp2[i]]['pos_y'],  t_cp[cp2[i]]['pos_z']

            pos_x = np.insert(pos_x, [0, -1], [pos_x_cp1, pos_x_cp2])
            pos_y = np.insert(pos_y, [0, -1], [pos_y_cp1, pos_y_cp2])
            pos_z = np.insert(pos_z, [0, -1], [pos_z_cp1, pos_z_cp2])
            
            # calculate the median values in cartesian space and then convert to ra, dec
            median_x, median_y, median_z = np.median(pos_x), np.median(pos_y), np.median(pos_z) 
            c = SkyCoord(x=median_x, y=median_y, z=median_z, unit='Mpc', representation_type='cartesian')    
            ra, dec, dist = c.fk5.ra, c.fk5.dec, c.fk5.distance
            
            ra_median_arr  = np.append( ra_median_arr, ra.value)
            dec_median_arr  = np.append( dec_median_arr, dec.value)
            dist_median_arr = np.append( dist_median_arr, dist.value)

        # convert distance to redshift
        Z_median_arr = np.array([])
        
        for dist in dist_median_arr:
            z = z_at_value(Planck18.comoving_distance, dist*u.Mpc, zmin= zmin, zmax= self.zmax)
            Z_median_arr = np.append(Z_median_arr, z)
        
        # delete column if it exists from previous runs
        if 'median_ra' in t.colnames :
            t.remove_column('median_ra')
        if 'median_dec' in t.colnames:
            t.remove_column('median_dec')
        if 'median_redshift' in t.colnames:
            t.remove_column('median_redshift')
        
        self.logger.info(f"Adding new column with filament meadian vals, len({len(Z_median_arr)}). Table has length: {len(t)}")
        t.add_column(ra_median_arr, name='median_ra')
        t.add_column(dec_median_arr, name='median_dec')
        t.add_column(Z_median_arr, name='median_redshift')
          
        t.write(f'{self.reformatted_filename}.fit', overwrite=True)
        self.logger.info("Saved table")

    def get_region_around_filament(self, extra_edges_crop: float = 20):
        """Method to get the region around a filament for inputting the values into eSASS tools
        Parameters
        ----------
        @zmin :: float
            the minimun redshift for 
        """
        self.extra_edges_crop = extra_edges_crop

        t = Table.read(f'{self.reformatted_filename}.fit')
        t_cp = Table.read(f'{self.base_name}_cp.fit')

        cp1, cp2 = t['cp1'], t['cp2']
        dist2one_end_arr, dist2other_end_arr, crop_around_fil_arr = np.array([]), np.array([]), np.array([])
        
        for i in range(len(t)):
            pos_x_cp1, pos_y_cp1, pos_z_cp1 = t_cp[cp1[i]]['pos_x'],  t_cp[cp1[i]]['pos_y'],  t_cp[cp1[i]]['pos_z']
            pos_x = np.array(t[i]['pos_x_arr'].split(',')).astype(float)
            pos_y = np.array(t[i]['pos_y_arr'].split(',')).astype(float)
            pos_z = np.array(t[i]['pos_z_arr'].split(',')).astype(float)
            pos_x_cp2, pos_y_cp2, pos_z_cp2 = t_cp[cp2[i]]['pos_x'],  t_cp[cp2[i]]['pos_y'],  t_cp[cp2[i]]['pos_z']

            pos_x = np.insert(pos_x, [0, -1], [pos_x_cp1, pos_x_cp2])
            pos_y = np.insert(pos_y, [0, -1], [pos_y_cp1, pos_y_cp2])
            pos_z = np.insert(pos_z, [0, -1], [pos_z_cp1, pos_z_cp2])
            
            # calculate the median values in cartesian space and then convert to ra, dec
            median_x, median_y, median_z = np.median(pos_x), np.median(pos_y), np.median(pos_z) 

            dist_one_end = np.sqrt( (pos_x_cp1-median_x)**2 + (pos_y_cp1-median_y)**2 + (pos_z_cp1-median_z)**2 )
            dist_other_end = np.sqrt( (pos_x_cp2-median_x)**2 + (pos_y_cp2-median_y)**2 + (pos_z_cp2-median_z)**2 )

            longer_end_of_filament = np.max([dist_one_end, dist_other_end])
            crop_size_Mpc = longer_end_of_filament + self.extra_edges_crop
            
            dist2one_end_arr  = np.append( dist2one_end_arr, dist_one_end)
            dist2other_end_arr  = np.append( dist2other_end_arr, dist_other_end)
            crop_around_fil_arr = np.append( crop_around_fil_arr, crop_size_Mpc)

        # delete column if it exists from previous runs
        if 'Dist_to_end1' in t.colnames :
            t.remove_column('Dist_to_end1')
        if 'Dist_to_end2' in t.colnames:
            t.remove_column('Dist_to_end2')
        if 'Crop_size_Mpc' in t.colnames:
            t.remove_column('Crop_size_Mpc')
        
        self.logger.info(f"Adding new column with filament crop vals, len({len(crop_around_fil_arr)}). Table has length: {len(t)}")
        t.add_column(dist2one_end_arr, name='Dist_to_end1')
        t.add_column(dist2other_end_arr, name='Dist_to_end2')
        t.add_column(crop_around_fil_arr, name='Crop_size_Mpc')
          
        t.write(f'{self.reformatted_filename}.fit', overwrite=True)
        self.logger.info("Saved table")
    
    def add_skytile_info(self, shape: str = 'box', size: float = None, sizein: int = 0, npix: int = 101, selection_cuts: list = [True, 'all']):
        """Method to add the skytile information into the filament table

        Parameters
        ----------
        @ra, dec :: float (unit: degrees)
            the ra and dec around which the skytiles are found
        @shape :: string (default: 'annulus')
        @size :: int (unit: degree)
            size to check for skytiles around input ra and dec
        @sizein :: int
            keyword used if 'shape' is set to 'annulus'
        @extra_edges_crop :: float (unit: Mpc)
            distance around the filament to crop
        @selection_cuts : list
            decides wether to find sky tiles for all filaments or only the selected ones
            - [True, 'all'] == all cuts applied
            - [True, 'length' ('angle')] == only length (orientation) cut applied
        """
        self.shape = shape
        t = QTable.read(f'{self.reformatted_filename}.fit')

        # decides for which filaments to find the skytiles
        if selection_cuts[0] and selection_cuts[1] == 'all':
            selected_entires = np.where(self.select_lengths & self.orientation_cut & (t['Ownership']=='DE'))[0]
        if selection_cuts[0] and selection_cuts[1] == 'length':
            selected_entires = np.where(self.select_lengths & (t['Ownership']=='DE'))[0]
        if selection_cuts[0] and selection_cuts[1] == 'angle':
            selected_entires = np.where(self.orientation_cut & (t['Ownership']=='DE'))[0]
        if not selection_cuts[0]:
            selected_entires = np.arange(len(t))
        self.selected_entires = selected_entires

        tiles_arr, percentage_arr = [], []
        
        for i in range(len(t)):
            if i in selected_entires:
                ra, dec, z = t[i]['median_ra']+self.ra_c, t[i]['median_dec']+self.dec_c, t[i]['median_redshift']
                crop_size_Mpc = t[i]['Crop_size_Mpc']

                if size is None:
                    size = (crop_size_Mpc*1e3*u.kpc)/ (self.cosmo.kpc_proper_per_arcmin(z))
                    size = size.to(u.degree).value 
                    if self.verbrose == 1:
                        self.logger.info(f"Size croping around the filament {size:.2f}")
                self.size = size
                self.sizein = sizein 
                self.npix = npix
                
                myimg = np.zeros((npix, npix))
                myhdu = fits.PrimaryHDU(myimg)
                myhdu.header['CRPIX1'] = npix / 2
                myhdu.header['CRPIX2'] = npix / 2
                myhdu.header['CDELT1'] = size / npix
                myhdu.header['CDELT2'] = size / npix
                myhdu.header['CRVAL1'] = ra
                myhdu.header['CRVAL2'] = dec
                myhdu.header['CTYPE1'] = 'RA---SIN'
                myhdu.header['CTYPE2'] = 'DEC--SIN'
                myhdu.header['CUNIT1'] = 'deg'
                myhdu.header['CUNIT2'] = 'deg'
                mywcs = wcs.WCS(myhdu.header)
                tiles, percentage = self.get_sky_tiles(mywcs, myhdu)
                
                tiles_arr.append(', '.join(map(str, list(tiles))))
                percentage_arr.append(', '.join(map(str, list(percentage))))
            else:
                tiles_arr.append('None')
                percentage_arr.append('None')

        if 'Sky_tile_no' in t.colnames:
            t.remove_column('Sky_tile_no')
        if 'percentage_in_sky_tile' in t.colnames:
            t.remove_column('percentage_in_sky_tile')

        self.logger.info(f"Adding new column with sky tile no and percentage, len({len(tiles_arr)}). Table has length: {len(t)}")
        t.add_column(tiles_arr, name='Sky_tile_no')
        t.add_column(percentage_arr, name='percentage_in_sky_tile')

        t.write(f'{self.reformatted_filename}.fit', overwrite=True)
        self.logger.info(" Modified saved table")

    def get_sky_tiles(self, mywcs, myhdu):
        """Method to get the eRosita skytiles within which the filament information is located
        @mywcs :: object
            object creating by the WCS (world coordinate system) used for high-level API
        @myhdu :: 
            the fits file header input
        Notes
        -----
        The 'RA_MIN', 'RA_MAX', 'DE_MIN', and 'DE_MAX' columns in SKYMAPS_NewOwner.fits are incorrect, so we only use RA_CEN and DE_CEN.
        """
        myimg = np.zeros((self.npix, self.npix))
        
        yw, xw = myimg.shape
        yc = np.array([np.full((xw,), y + 0.5) for y in range(xw)])
        xc = yc.T
        if self.shape == 'annulus':
            for i in range(xw):
                for j in range(xw):
                    if (i - xw / 2) ** 2 + (j - xw / 2) ** 2 > (xw / 2) ** 2 or (i - xw / 2) ** 2 + (j - xw / 2) ** 2 < (
                            xw / 2 * self.sizein / self.size) ** 2:
                        xc[i, j] = np.nan
                        yc[i, j] = np.nan

        # sky tile data 
        filename = os.path.join(self.save_data_dir, "SKYMAPS_052022.fits")
        skymap = fits.open(filename)
        allsky = skymap['SMAPS'].data
        de_sky = allsky[allsky['OWNER'] != 1] 
        
        # information used to find skytiles
        tile_nr = de_sky['SRVMAP']
        tile_ra_cen = de_sky['RA_CEN']
        tile_dec_cen = de_sky['DE_CEN']
        skymap.close()

        # gets the tile number
        tiles, percentage = np.array([]), np.array([])
        for this_tile in range(len(tile_nr)):
            # assuming each sky tile is 3.6 degree by side
            maxpix = 3.6 / mywcs.wcs.cdelt[0] / 2
            this_tile_ra_cen = tile_ra_cen[this_tile]
            this_tile_dec_cen = tile_dec_cen[this_tile]
            
            myhdu.header['CRVAL1'] = this_tile_ra_cen
            myhdu.header['CRVAL2'] = this_tile_dec_cen
            this_tile_imgwcs = wcs.WCS(myhdu.header)

            # convert a set of pixel coordinates into a SkyCoord coordinate.
            coords = wcs.utils.pixel_to_skycoord(xc, yc, mywcs)
            ras = coords.ra.to_value(u.deg)
            decs = coords.dec.to_value(u.deg)

            # transforms pixel coordinates to world coordinates
            pix_x, pix_y = this_tile_imgwcs.all_world2pix(ras, decs, 0) - mywcs.wcs.crpix[1]
            whichin = ((np.abs(pix_x) <= maxpix) & (np.abs(pix_y) <= maxpix))
            
            if len(np.where(whichin)[0]) > 0:
                this_tile_nr, occupied_src_area = tile_nr[this_tile], 100 * len(np.where(whichin)[0]) / (len(yc) ** 2 - len(yc[np.isnan(yc)]))
            else:
                this_tile_nr, occupied_src_area = None, None
            
            if this_tile_nr is not None:
                tiles = np.append(tiles, this_tile_nr)
                percentage = np.append(percentage, occupied_src_area)

        sorted_idx = percentage.argsort()
        percentage = percentage[sorted_idx[::-1]]
        tiles = tiles[sorted_idx[::-1]]
        tiles = np.array([int(tile) for tile in tiles])
        return tiles, percentage
    
    def give_name(self, fil_id: int = None, fil_len: float = None, nfil_segments: int = None, rcrop: float = None, median_z: float = None):
        """Method to give every filament and it's directory a name
        Return
        ------
        @filament_name :: str
            the name of the individual filament
        @directory_name :: str
            the name of the directory in which the filaments are saved
        """ 
        if np.any([fil_id, fil_len, nfil_segments, rcrop]) == None:
            fil_id = self.fil_id
            fil_len = self.fil_len
            nfil_segments = self.nfil_segments
            rcrop = self.rcrop_whole_filament
            median_z = self.median_z

        filament_name = f"{self.section_keyword}id{fil_id}_len_{fil_len:.2f}Mpc_{nfil_segments}seg_{rcrop:.2f}deg_z{median_z:.3f}"
        directory_name = f"{self.data_set}_{self.smoothing_density_f}_s{self.persistence:.1f}_{self.smoothing_skeleton}"
        return directory_name, filament_name

    def get_data_products(self, fil_id: int = None, proc_v_folder: str = None, clobber: bool = False, filament_name: str = None, directory_name: str = None, flag="0xc00fff30", pattern=None, bandEs=None, crop_size_Mpc: float = 40, rcrop=None, skytiles=None, rebin=None, cut_out: str = 'box', median_ra: float = None, median_dec: float = None) -> None:
        """
        Function sets up several parameters for generating eROSITA data products, 
        - creates the folder structure required by the analysis
        - runs image creation
        - exposure map calculation (vignetted and not vignetted)
        - computes the cheese mask

        Parameters
        ----------
        @fil_id :: int (default: None)
            the unique filament number assigned to every filament
        @data_location :: string (default: None)
            the location where the data products are stored
        @proc_v_folder :: str
            the version of the processing used to generate event files by the eSASS team
        @clobber :: bool
            bool decides wether or not the files need to be recreated
        @filament_name :: str
            the name of the filament, if not given used the method 'give_name()'
        @directory_name :: str
            the name of the directory defined by DisPerSE parameters, if not given used the method 'give_name()'
        @flag :: str (hexadecimal)
            string to filter events based on the value of FLAG
        @pattern :: int (from 1 to 15, default used is 15)
            filter model (based on values of PAT_TYP) for selecting good/bad patterns
        @bandEs :: 1D-array of tuples
            the energy bands within which the images are generated
        @crop_size_Mpc :: float (unit: Mpc)
            the region size for cropping around the filament
        @rcrop :: float (unit: degree)
            the region around the "median" of the filament that is cropped (same as crop_size_Mpc)
        @rebin :: 
            parameter specifies the integer number of virtual X/Y pixels to fit into a single image pixel
        @cut_out :: str (default: 'box')
            the shape of the image to be cut out from eSASS
        @median_ra, median_dec :: float (default: None)
            the median (ra, dec) of the filament in equatorial coordinates 
        """
        t = Table.read(f'{self.reformatted_filename}.fit')
        
        if fil_id is None:
            raise Warning("You need to pass a 'fil_id': filament ID number")
        self.fil_id = fil_id
        self.logger.info(f"{fil_id=}")
        
        self.fil_len = t[fil_id]['Fil_lengths_Mpc']
        self.nfil_segments = int(t[fil_id]['n_samp'])
        
        # decides the version of the eSASS pipeline 
        if proc_v_folder is None:
            proc_v_folder = "c020"
            self.logger.info(f"'proc_v_folder' default set to {proc_v_folder}")

        if clobber == True:
            _clobber='yes'
        else:
            _clobber='no'       
        
        if pattern is None:
            pattern="15"
            self.logger.info(f"'pattern' not set; setting it to the default value of '{pattern}'")

        # decide energy band for processing
        if bandEs is None:
            bandEs=( (200, 2300), (200, 600), (600, 1100), (1100, 1600), (1600,2200),\
                (2200, 3500), (3500, 5000), (5000, 7000))
            self.logger.info(r"'bandEs' not set; setting it to the default value of '{0}'. \
                Remember that units are eV. Also note that you can also pass tuples,\
                e.g. bandEs=( (200, 600), (600, 1100), (1100, 1600), (1600,2200),\
                (2200, 3500), (3500, 5000), (5000, 7000))".format(bandEs))

        if np.min(bandEs) < 10:
            raise Warning("I don't think you remembered to pass 'bandEs' in eV")
        
        if skytiles is None:
            # get sky tile number of the concerned object from the table  
            skytiles_list = t[fil_id]['Sky_tile_no']
            if skytiles_list is np.ma.masked:
                self.logger.info(" This array is masked")
                skytiles_list = skytiles_list.data
            skytiles = np.array(skytiles_list.split(', ')).astype(int)
            self.logger.info(f" Number of computed skytile(s) is/are {len(skytiles)}")
            

        if crop_size_Mpc is None:
            crop_size_Mpc = t[fil_id]['Crop_size_Mpc']
        self.crop_size_Mpc = crop_size_Mpc

        # radial crop of image
        if rcrop is None:
            rcrop = self.get_rcrop(t[self.fil_id]['median_redshift'])
            self.logger.info(f"'rcrop' set to {rcrop: .2f} degrees") 
        self.rcrop = rcrop
        self.median_z = t[self.fil_id]['median_redshift']

        # radial crop of the "entire" filament + 20 Mpc
        self.rcrop_whole_filament = self.get_rcrop(t[self.fil_id]['median_redshift'], crop_size_Mpc=t[self.fil_id]['Crop_size_Mpc'])

        if rebin is None:
            rebin = 80
        min_rebin = int((80*self.rcrop*3600)/(18000*4))

        if rebin <= min_rebin:
            rebin = int(min_rebin + 1)
        
        scale_size_arcsec = (rebin*4.3/80)

        size = self.rcrop*3600//scale_size_arcsec 
        self.size = int(size)
        
        if filament_name is None and directory_name is None: 
            directory_name, filament_name=self.give_name()
        self.directory_name = directory_name
        self.filament_name = filament_name

        # save data on the chosen extended source
        name_folder = f"{self.data_location}/{directory_name}/{filament_name}"
        if not os.path.exists(name_folder):
            os.makedirs(name_folder)

        # to store the eventfile list for every filament
        eventfiles_folder=f"{self.data_location}/{directory_name}/eventfiles"
        if not os.path.exists(eventfiles_folder):
            os.makedirs(eventfiles_folder)

        inputs_folder=f"{name_folder}/inputs"
        if not os.path.exists(inputs_folder):
            os.makedirs(inputs_folder)

        outputs_folder=f"{name_folder}/fits"
        if not os.path.exists(outputs_folder):
            os.makedirs(outputs_folder)

        rescaled_folder = f"{name_folder}/rescaled_inputs"
        if not os.path.exists(rescaled_folder):
            os.makedirs(rescaled_folder)

        commands_folder = f"{name_folder}/eSASS_commands"
        if not os.path.exists(commands_folder):
            os.makedirs(commands_folder)

        if median_ra == None and median_dec == None: 
            median_ra = t[self.fil_id]['median_ra']+self.ra_c
            median_dec = t[self.fil_id]['median_dec']+self.dec_c

            c = SkyCoord(median_ra, median_dec, frame='icrs', unit=u.deg)
            self.median_ra, self.median_dec = c.ra.value, c.dec.value

        # decides wether the cropped region is a box or a circle
        if cut_out == 'circle':
            region_choice = f"fk5;circle({self.median_ra:.5f},{self.median_dec:.2f},{self.rcrop:.2f})"
        
        # make sure the rcrop covers > 20 Mpc around filament ends and spine 
        elif cut_out == 'box':
            region_choice = f"fk5;box({self.median_ra:.5f},{self.median_dec:.2f},{self.rcrop:.2f},{self.rcrop:.2f},0)"
            self.logger.info(f' {cut_out} sides are {self.rcrop:.2f} degree, image has {self.size} pixels')

        self.region_file = f"{inputs_folder}/fil.reg"
        save_region_fil = open(self.region_file, 'w')
        save_region_fil.write(f"{region_choice}\n")
        save_region_fil.write(f"\n")
        save_region_fil.close()

        self.redshift = t[fil_id]['median_redshift']
        # public class variables
        self.skytiles = skytiles
        self.proc_v_folder=proc_v_folder
        self.bandEs=bandEs
        self.crop_size_Mpc = crop_size_Mpc
        self.clobber = clobber
        self.rebin = rebin
        self.inputs_folder=inputs_folder
        self.outputs_folder=outputs_folder
        self.eventfiles_folder = eventfiles_folder
        self.rescaled_folder = rescaled_folder
        self.commands_folder = commands_folder
        self.cut_out = cut_out

        # private and protected class variables        
        self._clobber=_clobber
        self.__flag=flag
        self.__pattern=pattern
        self.__bkg=1e-3
        
    
    def get_rcrop(self, median_z, crop_size_Mpc: float = None):
        "Function to get the size in arcsec for cropping the eventfiles"
        if crop_size_Mpc is None:
            crop_size_Mpc = self.crop_size_Mpc
        rcrop_arcmin = (crop_size_Mpc*1e3*u.kpc)/ (self.cosmo.kpc_proper_per_arcmin(median_z))
                
        rcrop = rcrop_arcmin.to(u.deg).value
        return rcrop    

    def create_eventfile_list(self):
        """Function to create a list of eventfiles belonging to the filament of concern
        
        Returns
        -------
        column 'FLAG_eventfile_list' within filament file ::
            the purpose of this column is to add a flag for the filaments in the erass sky
        """
        t = Table.read(f'{self.reformatted_filename}.fit')
        
        flag_filaments = np.zeros(len(t))

        get_masked_idx = [np.ma.isMaskedArray(tile) for tile in t['Sky_tile_no']]
        get_masked_idx = np.logical_not(get_masked_idx)
        
        fil_ids_selected = np.where(get_masked_idx & (t['Sky_tile_no' ] != 'None') & 
                                  self.select_lengths & 
                                  self.orientation_cut          
                                 )[0]
        # get all the event files
        for i in range(len(t)):
            for fil_id in fil_ids_selected:
                skytiles = t[fil_id]['Sky_tile_no']
                # get filament name
                crop = t[fil_id]['Crop_size_Mpc']
                directory_name, filament_name = self.give_name(
                    fil_id = fil_id, 
                    fil_len = t[fil_id]['Fil_lengths_Mpc'], 
                    nfil_segments = int(t[fil_id]['n_samp']), 
                    rcrop = self.get_rcrop(t[fil_id]['median_redshift'], crop_size_Mpc=crop),
                    median_z =  t[fil_id]['median_redshift']
                    )
                
                if not os.path.exists(f'{self.data_location}/{directory_name}/eventfiles'):
                    os.makedirs(f'{self.data_location}/{directory_name}/eventfiles')

                # get skytiles
                skytiles = np.array(skytiles.split(',')).astype(int)
                
                all_filenames = []
                for tile in skytiles:
                    skytile_str = str(tile).zfill(6)
                    filenames = glob.glob(f"{self.eRASS4_dir}/{skytile_str}/c020/*_EventList*fits*")[0] 
                    if os.path.isfile(filenames):
                        all_filenames = np.append(all_filenames, filenames)

                if len(all_filenames) > 0:
                    np.savetxt(f"{self.data_location}/{directory_name}/eventfiles/{filament_name}.txt", all_filenames, fmt='%s')
                    flag_filaments[fil_id] = 1
                
        if 'FLAG_eventfile_list' in t.colnames:
            t.remove_column('FLAG_eventfile_list')
        
        self.logger.info(f"Adding new column to flag filaments, len({len(flag_filaments)}). Table has length: {len(t)}")
        t.add_column(flag_filaments, name='FLAG_eventfile_list')
        
        t.write(f'{self.reformatted_filename}.fit', overwrite=True)
        self.logger.info(" Modified saved table")

    def generate_filament_points(self):
        """Function to generate the array of ra, dec of the filament points on the filament
        """
        # get the image centers
        t_extras = Table.read(f'{self.reformatted_extras_filename}.fit')
        ids = np.where(t_extras['index'] == self.fil_id+1)
        
        ra, dec = t_extras[ids]['RA']+self.ra_c, t_extras[ids]['DEC']+self.dec_c
        c = SkyCoord(ra, dec, frame='icrs', unit='degree')
        ra, dec = c.ra.value, c.dec.value
        coord = [(r, d) for r, d in zip(ra, dec)]       
        return coord

    def generate_image_centers(self) -> None:
        """
        """
        coord = self.generate_filament_points()
        ra_mid_coord, dec_mid_coord = np.array([]), np.array([])
        for i in range(len(coord)-1):
            ra1, dec1 = coord[i]
            ra2, dec2 = coord[i+1]
            coord1 = SkyCoord(ra1*u.deg, dec1*u.deg, frame='icrs')
            coord2 = SkyCoord(ra2*u.deg, dec2*u.deg, frame='icrs')
            pa = coord1.position_angle(coord2)
            sep = coord1.separation(coord2)
            midpoint_coord = coord1.directional_offset_by(pa, sep/2)  
            
            ra_mid_coord = np.append(ra_mid_coord, midpoint_coord.ra.value)
            dec_mid_coord = np.append(dec_mid_coord, midpoint_coord.dec.value)
        mid_coord = np.array([ra_mid_coord, dec_mid_coord])
        np.savetxt(f"{self.inputs_folder}/image_centers.txt", np.array(mid_coord))
        return

    def generate_filament_point_region_files(self):
        """Function to generate the region files around each filament point
        """
        self.generate_image_centers()
        coord = np.loadtxt(f"{self.inputs_folder}/image_centers.txt")
        
        for i in range(len(coord[0])):
            ra, dec = coord[0, i], coord[1, i]
            
            # center the image on every filament segment midpoint
            region_choice = f"fk5;box({ra:.5f},{dec:.2f},{2*self.rcrop:.3f},{2*self.rcrop:.3f},0)"
            region_file = f"{self.inputs_folder}/fil_{i}.reg"
            save_region_fil = open(region_file, 'w')
            save_region_fil.write(f"{region_choice}\n")
            save_region_fil.close()

    def get_selected_filament_ids(self):
        """Function to get the relavant filament ids for which the data products are generated
        """
        t = Table.read(f'{self.reformatted_filename}.fit')
        
        self.logger.info(f"masked entries={np.ma.count_masked(t['Sky_tile_no' ])}")
        get_masked_idx = [np.ma.isMaskedArray(tile) for tile in t['Sky_tile_no']]
        get_masked_idx = np.logical_not(get_masked_idx)
        
        fil_ids_selected = np.where(get_masked_idx & (t['Sky_tile_no' ] != 'None') & 
                                  self.select_lengths & 
                                  self.orientation_cut          
                                 )[0]
        return fil_ids_selected    
        
    def clean_evt(self, clobber=None):
        """Function cleans the EventFiles and crops them around the given position
        """ 
        emin=np.min(self.bandEs)/1000
        emax=np.max(self.bandEs)/1000
        
        # generates region files used for creating cut outs
        self.generate_filament_point_region_files()

        if clobber is None:
            clobber_key = self._clobber 
        elif clobber is False:
            clobber_key = 'no'
        elif clobber is True:
            clobber_key = 'yes'

        # executes evtool
        for i in range(self.nfil_segments):
            out_evt_file=f"{self.inputs_folder}/clean_evt_{i}.fits.gz"

            if not os.path.isfile(out_evt_file) or clobber_key == 'yes':
                clean_evt_cmd_filename = f"{self.commands_folder}/clean_evt_{i}"
                clean_evt_cmd = open(clean_evt_cmd_filename, 'w')
                cmd="evtool" +  \
                f" eventfiles=@{self.eventfiles_folder}/{self.filament_name}.txt" +\
                f" outfile={out_evt_file}" + \
                f" emin={emin:.1f}" + \
                f" emax={emax:.1f}" + \
                f" region={self.inputs_folder}/fil_{i}.reg" + \
                 " image=no" + \
                 " telid=\"1 2 3 4 6\"" + \
                f" clobber=\"{clobber_key}\""
                
                clean_evt_cmd.write(f"{cmd} \n")
                clean_evt_cmd.write('\n')
                clean_evt_cmd.close()
                subprocess.check_call(["bash", clean_evt_cmd_filename])
                
                self.logger.info(f'Cleaned event file? {os.path.isfile(out_evt_file)}')
            else:
                self.logger.info(f"EventFiles {out_evt_file} already exist")
        self.out_evt_files = [f"{self.inputs_folder}/clean_evt_{i}.fits.gz" for i in range(self.nfil_segments)]
        self.clean_evt_cmds = [f"{self.commands_folder}/clean_evt_{i}" for i in range(self.nfil_segments)]
        self.clobber_clean_evtfiles = clobber_key

    def make_images(self, clobber=None):
        """
        '_make_images_' method makes the images from the cleaned EventFiles in the desired energy bands
        """
        if clobber is None:
            clobber_key = self._clobber 
        elif clobber is False:
            clobber_key = 'no'
        elif clobber is True:
            clobber_key = 'yes'

        bandEs = self.bandEs
        out_ima_files = np.array([])
        coord = np.loadtxt(f"{self.inputs_folder}/image_centers.txt")
        for i in range(len(coord[0])):
            out_evt_file = self.out_evt_files[i]
            ra, dec = coord[0, i], coord[1, i]
            self.logger.info(f"ra={ra:.2f}, dec={dec:2f}")
            
            xval, yval = cen_pos(ra = ra, dec = dec, evt_file = out_evt_file)
            self.logger.info(f"xval={xval:.2f}, yval={yval:2f}")
            for band in bandEs:
                bandname = f'{band[0] / 1000 :.1f}_{band[1] / 1000 :.1f}'
                out_ima_file = f"{self.inputs_folder}/ima_{bandname}_{i}.fits.gz"
                if not os.path.isfile(out_ima_file) or clobber_key == 'yes': 
                    make_img_cmd_filename = f"{self.commands_folder}/make_images_{bandname}_{i}"
                    make_img_cmd = open(make_img_cmd_filename, 'w')
                    cmd = "evtool" +\
                           f" eventfiles={out_evt_file}" +\
                           f" outfile={out_ima_file}" +\
                           f" emin={band[0] / 1000 :.1f}" +\
                           f" emax={band[1] / 1000 :.1f}" +\
                            " image=yes" +\
                            " events=no" +\
                           f" center_position=\"{xval:.2f} {yval:.2f}\"" + \
                           f" clobber=\"{clobber_key}\""
                           
                           
                    make_img_cmd.write(f"{cmd} \n")
                    make_img_cmd.write('\n')
                    make_img_cmd.close()
                    subprocess.check_call(["bash", make_img_cmd_filename])
                    self.logger.info(f'Image created? {os.path.isfile(out_ima_file)}')

                    out_ima_files = np.append(out_ima_files, out_ima_file)
                    self.make_img_cmd = make_img_cmd
                else:
                    self.logger.info(f"Image {out_evt_file} already exist")
                    out_ima_files = np.append(out_ima_files, out_ima_file)

        image_names_path = f"{self.inputs_folder}/all_filament_image_names.txt"
        np.savetxt(image_names_path, out_ima_files,  fmt='%s')
        self.out_ima_files = image_names_path
        self.clobber_make_imgs = clobber_key
        
    def mask_clusters_point_srcs(self, src_catalog: str = None, band: tuple=(200, 2300), clobber=None):
        """
        Method to mask clusters and point sources along the line of sight of the filament
        Parameters
        ----------
        src_catalog :: str
            the name of the point course and clusters catalog
        band :: tuple (default: (200, 2300); units: eV)
            the default image used for creating masks
        clobber :: bool (default: None)
            if user would like to regenerate (not regenerate) mask files, can be set to True (False)
        """
        if clobber is None:
            clobber_key = self._clobber 
        elif clobber is False:
            clobber_key = 'no'
        elif clobber is True:
            clobber_key = 'yes'

        if src_catalog is None:
            src_catalog = f"{self.base_dir}/data/Stacking_filaments/masklist.fits"
        self.src_catalog = src_catalog

        bandname = f'{band[0] / 1000 :.1f}_{band[1] / 1000 :.1f}'
                
        out_mask_files = np.array([])
        coord = np.loadtxt(f"{self.inputs_folder}/image_centers.txt")
        for i in range(len(coord[0])):
            output = f'{self.inputs_folder}/mask_{i}.fits.gz'
            
            if clobber_key == 'yes' and os.path.isfile(output):
                os.remove(output)

            if not os.path.isfile(output):
                image = f'{self.inputs_folder}/ima_{bandname}_{i}.fits.gz'
                hdulist = fits.open(image)
                ima_map = hdulist[0].data
                prihdr = hdulist[0].header
                pix2arcmin = prihdr['CDELT2'] * 60  # arcmin
                xsize, ysize = ima_map.shape
                mask = np.copy(ima_map) * 0 + 1.

                w = wcs.WCS(prihdr, relax=False)
                sky = w.pixel_to_world(xsize/2, ysize/2)
                ra = sky.ra.deg
                dec = sky.dec.deg

                data_src = fits.open(src_catalog)[1].data
                # Make a cut on dec to reduce the number of distances to be computed
                # the cut on ra is too complicated, ignored...
                data_src = data_src[data_src['dec_corr'] < dec + ysize *
        pix2arcmin / 60 * 1.2]
                data_src = data_src[data_src['dec_corr'] > dec - ysize *
        pix2arcmin / 60 * 1.2]

                # Select only in-image sources
                ra_src = data_src['ra_corr']
                dec_src = data_src['dec_corr']
                dist = astropy_dist(ra_src, dec_src, ra, dec)
                ii = np.where(dist < 1.2 * np.max((xsize, ysize)) / 2 * pix2arcmin / 60)
                data_src = data_src[ii]

                ra_src = data_src['ra_corr']
                dec_src = data_src['dec_corr']
                ext_src = data_src['src_ext']

                ii = np.where(ext_src < 30)
                ext_src[ii] = 30

                x = np.arange(ysize)
                y = np.arange(xsize)
                for j in range(len(ra_src)):
                    pixim = w.all_world2pix([[float(ra_src[j]), float(dec_src[j])]], 0)
                    xp = pixim[0][0]
                    yp = pixim[0][1]
                    rad = ext_src[j] / 60
                    rho = circle(x - xp, y - yp) * pix2arcmin
                    ii = np.where(rho <= rad)
                    if len(ii)>0:
                        mask[ii] = 0

                hdu = fits.PrimaryHDU(mask, header=prihdr)
                hdulist = fits.HDUList([hdu])
                hdulist.writeto(output, overwrite=True)

            out_mask_files = np.append(out_mask_files, output)
        self.out_mask_files = out_mask_files
        self.clobber_masks = clobber_key

    def get_exposures(self, clobber=None):
        """Function to get the exposure maps for every filament
        """
        if clobber is None:
            clobber_key = self._clobber 
        elif clobber is False:
            clobber_key = 'no'
        elif clobber is True:
            clobber_key = 'yes'

        out_ima_files = np.loadtxt(self.out_ima_files, dtype=str)
        out_exp_vig_files = np.array([])

        # iterating over the event files and the bandnames
        coord = np.loadtxt(f"{self.inputs_folder}/image_centers.txt")
        for i in range(len(coord[0])):
            out_evt_file = self.out_evt_files[i]
            for band in self.bandEs:
                # generates the bandnames in units of keV 
                bandname = f'{band[0] / 1000 :.1f}_{band[1] / 1000 :.1f}'
                out_exp_vig_file = f"{self.inputs_folder}/exp_vig_{bandname}_{i}.fits.gz"
                out_ima_file = f"{self.inputs_folder}/ima_{bandname}_{i}.fits.gz"
                # check first if image exists
                if out_ima_file in out_ima_files:
                    self.logger.info("Image exists :) ")

                if clobber_key == 'yes' and os.path.isfile(out_exp_vig_file):
                    os.remove(out_exp_vig_file)
                    
                if not ((clobber_key == 'no') and (os.path.isfile(out_exp_vig_file))):
                    make_exp_cmd_filename = f"{self.commands_folder}/make_exp_vig_{bandname}_{i}"
                    make_exp_vig_cmd = open(make_exp_cmd_filename, 'w')
                    cmd = "expmap" + \
                          f" inputdatasets={out_evt_file}" + \
                          f" templateimage={out_ima_file}" + \
                          f" mergedmaps={out_exp_vig_file}" + \
                          f" emin={band[0] / 1000 :.1f}" + \
                          f" emax={band[1] / 1000 :.1f}" + \
                          f" withsinglemaps=no" + \
                          f" withdetmap=yes" + \
                          f" withmergedmaps=yes" + \
                          f" gtitype=GTI" + \
                          f" withvignetting=yes"
                    make_exp_vig_cmd.write(f"{cmd}\n")
                    make_exp_vig_cmd.write('\n')

                    subprocess.check_call(["bash", make_exp_cmd_filename])
                    self.logger.info(f'Exposure map created? {os.path.isfile(out_exp_vig_file)}')

                    out_exp_vig_files = np.append(out_exp_vig_files, out_exp_vig_file)
                    self.make_exp_vig_cmd = make_exp_vig_cmd
                else:
                    self.logger.info(f"Image {out_evt_file} already exist")
                    out_exp_vig_files = np.append(out_exp_vig_files, out_exp_vig_file)
        
        exp_vig_names_path = f"{self.inputs_folder}/all_filament_exp_vig_names.txt"
        np.savetxt(exp_vig_names_path, out_exp_vig_files,  fmt='%s')
        self.out_exp_vig_files = exp_vig_names_path
        self.clobber_exposur_maps = clobber_key
