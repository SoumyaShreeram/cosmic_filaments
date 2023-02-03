#!/usr/bin/pythons
"""
stackimages.py 

File owns the class to stack images of Cluster & ProcessCluster objects.

Author: Soumya Shreeram
Email: shreeram@mpe.mpg.de
Date created: 8th Feb 2022
"""

import numpy as np 
import logging
import random as rand
import os
import subprocess
import glob
import pandas as pd

# astropy modules
import astropy.units as u
import astropy.io.fits as fits
from astropy.table import Table

# plotting imports
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# locally written modules
from .Cluster import Cluster
from .Events2Files import Events2Files
from .DoPlot import DoPlot
from .utils import write_fits_file, set_labels


class StackImages:
    """Representation of stacking cluster images

    The stacking process involves the following steps:
    1. generation of images, exposure maps, masks
    2. correction of the exposure maps to account for redshift dimming
    3. saving the outputted information into a fits files, which is reprocessed by the 'ciao' software
    4. reloading the outputted files from 'ciao' and stacking this
    
    Step 4. is the main goal of this class. 

    Notes
    -----
    You need the following files in the local data directory:
    psf_prof.dat : 
        Required for generating masks for point sources

    pre_cwg_e1_c946_20210121-matched-mcxc-act-spt.fits:
        eRASS1 cross-matched cluster catalog created by Ang Liu.
        This is relevant only if choosing eRASS1 in the `survey='eRASS1'` keyword. 

    Examples
    --------
    Shown in notebook 03_Stacking_clusters.ipynb in the accompanying code directory.
    Refer to repository link: github.com/

    Methods
    -------
    self.select_objects() : defines cuts on clusters in mass and redshift
    self.process_clusters() : generates cluster images, exposure maps, masks
    self.process_exposure_maps() : corrects exp maps for redshift dimming
    self.save_maps_info() : saves (.fits) the file paths/useful info outputted from processing 
    self.reload_and_stack() : relaods the fiels generated with ciao and stacks them
    """
    def __init__(self, base_cluster: object = None, survey: str = None, proc_v_folder: str = None, cluster_catalog: str ='', stack_data_location: str = None, base_dir: str = None, 
        image_dir: str = None) -> None:
        """Initialied with a cluster object is created

        Paratemers
        ----------
        base_cluster :: object Cluster
            Object to define the base_cluster at which we stack the objects
        survey :: string
            Defines the survey used for creating cluster images, hence defining the directory location   
        proc_v_folder :: str; default set to 'c947'
            Specifies the version of the eSASS catalogue to be used
        """

        logging.basicConfig(level = logging.INFO)
        self.logger = logging.getLogger(type(self).__name__)
        
        if base_cluster is None:
            raise Warning("'base_cluster' not set")
        
        self.data_location = base_cluster.data_location
        if not os.path.isdir(self.data_location):
            self.logger.info(f"{self.data_location} does not exist!")

        self.eRASS1_dir = base_cluster.eRASS1_dir

        # function sets the survey and cluster/source catalog path
        self.set_survey(survey)

        # setting the directory structures
        if proc_v_folder is None:
            proc_v_folder = "c947"
            self.logger.info(f" Using default 'proc_v_folder={proc_v_folder}' option")

        if base_dir is None:
            base_dir = "/data53s/shreeram/Cluster_image_stacking"
            self.data_dir = f"{base_dir}/data_dir"

        if stack_data_location is None:
            stack_data_location = f"{self.data_dir}/stacking_test"
            self.logger.info(f" Cluster directories are stored at {stack_data_location}")

        if image_dir is None:
            image_dir = f"{base_dir}/images"

        self.base_cluster = base_cluster
        self.z_ref = self.base_cluster.redshift
        
        self.base_dir = base_dir
        self.stack_data_location = stack_data_location
        self.image_dir = image_dir
        
    def set_survey(self, survey):
        """
        Method sets the survey option and also provides the link to the appropriate cluster/source catalog

        Parameters
        ----------
        survey : str, defautl = None
            Decides which survey to use. Current options are
            - 'eFEDs_matched_eRASS': eRASS1 matched with eFEDs
            - eRASS1 
            - 'point_sources': Point source catalog 
        """
        if survey is None:
            survey = 'eRASS1'

        if survey == 'eFEDs':
            # cluster/groups
            # TODO: add the link to the eFEDs cluster catalog

            # point sources
            point_src_catalog = f'{self.data_location}/{survey}/final_e1_SourceCat1B_211229_poscorr_mpe_clean.fits'         
            self.z_key_pt_src = 'CTP_REDSHIFT''RA''DEC''Z'
            self.ra_key_pt_src, self.dec_key_pt_src = 'ERO_RA_CORR', 'ERO_DEC_CORR'
        
        elif survey == 'eRASS1':
            # clusters/groups
            self.logger.info(f" Using 'survey={survey}' option; this is v1 cluster_catalog by Ang Liu")
            cluster_catalog = f"{self.data_location}/{survey}/pre_cwg_e1_c946_20210121-matched-mcxc-act-spt.fits"
            self.z_keyword = 'z'

            # point sources
            point_src_catalog = f'{self.data_location}/{survey}/eromapper_merged_final_e1_SourceCat1B_211229_poscorr_mpe_clean_220410_catalog.fit'
            self.z_key_pt_src = 'Z'
            self.ra_key_pt_src, self.dec_key_pt_src = 'RA', 'DEC'
        
            if not os.path.isfile(cluster_catalog) or not os.path.isfile(point_src_catalog):
                raise Warning(f"Path given to '{cluster_catalog}' or '{point_src_catalog}' doesn't exist.")
            
        self.survey = survey 
        self.cluster_catalog = cluster_catalog

    def select_objects(self, mass_range: tuple = None, redshift_range: tuple = None,\
     num_objects: int = 10):
        """
        Function to decide which objects to select for stacking

        Parameters
        ----------
        mass_range :: tuple
            Decides the range of mass within which the objects are selected
        redshift_range :: tuple
            Decides the range of redshift within which the objects are selected
        num_objects :: int
            Decided how many objects to consider for stacking
        """
        self.num_objects = num_objects
        cluster_table = Table.read(self.cluster_catalog, format='fits')

        # selecting masses
        # ----------------
        if mass_range is None:
            mass_range = (2.3e+12, 1e14)
            self.logger.info(f"'mass_range' not set. Default values {mass_range}")
        else: 
            if mass_range[0] >= 1e13 and mass_range[1] <= 1e14:
                mass_category = 'group'
            elif mass_range[0] >= 1e14 and mass_range[1] <= 1e15:
                mass_category = 'clusters'
            else:
                raise Warning(f"Mass category not set!")    

        if self.survey == 'eRASS1':
            mass_range = [m/1e14 for m in mass_range]
            

        mass_constrain = (mass_range[0]<cluster_table['M500']) & (cluster_table['M500']<mass_range[1])
        cluster_table = cluster_table[mass_constrain]

        # selecting redshifts
        # -------------------
        if redshift_range is None:
            redshift_range = (0.03, 0.4)
            self.logger.info(f"'redshift_range' not set. Default values {redshift_range}")
        
        redshift_constrain = (redshift_range[0]<cluster_table[self.z_keyword]) & (cluster_table[self.z_keyword]<redshift_range[1])
        cluster_table = cluster_table[redshift_constrain]

        # redshift keyword
        if redshift_range[0] >= 0.028 and redshift_range[1] <= 0.4:
            redshift_keyword = 'low'
        elif redshift_range[0] >= 0.4 and redshift_range[1] <= 0.5:
            redshift_keyword = 'med'
        elif redshift_range[0] >= 0.5 and redshift_range[1] <= 0.6:
            redshift_keyword = 'high'
        else:
            redshift_keyword = 'all'

        self.mass_category = mass_category
        self.redshift_keyword = redshift_keyword

        # decide the objects to stack randomly
        # ------------------------------------
        if num_objects < len(cluster_table):
            rand_arr_idx = rand.choices( np.arange(len(cluster_table)), k=num_objects)
            cluster_table = cluster_table[rand_arr_idx]
            self.logger.info(f'stacking {len(cluster_table)} objects')
        else:
            self.logger.info(f"Number of groups/clusters/point srouces available is {len(cluster_table)}.")
            self.logger.info(f"'num_objects' {num_objects} exceeds this value.")
            self.logger.info(f"Selecting all {len(cluster_table)} objects for stacking... :o")

        self.selected_catalog_table = cluster_table
        self.mass_arr = self.selected_catalog_table['M500']
        self.z_arr = self.selected_catalog_table[self.z_keyword]
        self.ra_arr = self.selected_catalog_table['RA']
        self.dec_arr = self.selected_catalog_table['DEC']

    def process_clusters(self, resize_ima_to: int = 60, rebin: int = 20, 
        clobber: bool = False):
        """
        Function generates images, exp maps, masks for the clusters/groups chosen
        Parameters
        ----------
        resize_ima_to : int, default = 60
            The size of the final image that will be stacked
        rebin : int, default = 20
            `evtool`'s parameter `rebin` is used to produce higher resolution images for high-z objects
        clobber : bool, default set to False
            keyword decided if the images, exposure maps, and masks are regenerated
        """
        size_og_arr, cluster_names = [], []
        processed_cluster_objects = []
    
        for m, z, ra, dec in zip(self.mass_arr, self.z_arr, self.ra_arr, self.dec_arr):
            this_object = Cluster(mass=m, redshift=z, ra=ra, dec=dec, data_location=self.stack_data_location)
            self.logger.info(f"\nProcessing {this_object.name}")
            
            this_tuned_object = Events2Files(this_object)
            
            # generates and plots images, exp maps, and cheese mask
            this_tuned_object.do_reduction(clobber=clobber, cut_out='box', bandEs=((500, 2000),), 
                rebin=rebin)
            cluster_names.append(this_tuned_object.name)

            # calculate the scale factor
            ima_map = fits.open(this_tuned_object.out_ima_files[0])[0].data
            size_og = ima_map.shape[0]
            if resize_ima_to > size_og:
                self.logger.info(" you are asking to resize the image into a bigger size that the original one")
                resize_ima_to = size_og
                self.logger.info(f" resetting 'resize_ima_to={size_og}'")

            size_og_arr.append(size_og)
          
            processed_cluster_objects.append(this_tuned_object)
        
        self.resize_ima_to = resize_ima_to   
        self.size_og_arr = size_og_arr
        self.processed_cluster_objects = processed_cluster_objects
        self.cluster_names = cluster_names

    
    def process_exposure_maps(self, clobber: bool = True):
        """
        Function corrects exposure maps in preperation to stack them
        @clobber :: bool 
            param decides if the corrected exp maps need to be rewritten
        """
        energy_bands = self.processed_cluster_objects[0].bandEs

        for i in range(len(self.processed_cluster_objects)):
            out_exp_files = self.processed_cluster_objects[i].out_exp_files
            
            for band, out_exp_file in zip(energy_bands, out_exp_files):
                bandname = f'{band[0] / 1000 :.1f}_{band[1] / 1000 :.1f}'
                exp_map = fits.open(out_exp_file)[0].data
                
                # correct for redshift 
                z = self.processed_cluster_objects[i].redshift
                corr_factor = (1+z)**4/(1+self.z_ref)**4
                exp_map_corr = exp_map*corr_factor
                
                # write to a fits file in the "stacking_test directory"
                hdu = fits.PrimaryHDU(exp_map_corr)
                hdul = fits.HDUList([hdu])            
                this_cluster_exp_corr = f"{self.processed_cluster_objects[i].inputs_folder}/exp_corr_{bandname}.fits"
                
                if not os.path.isfile(this_cluster_exp_corr) and clobber:
                    hdul.writeto(this_cluster_exp_corr, overwrite=True)

                # save path location to reload later with ciao
                self.processed_cluster_objects[i].out_exp_corr_file = this_cluster_exp_corr
                
           
    def save_maps_info(self, overwrite: bool = True) -> None:
        """
        Function generates a fits file to save info about the generated maps
        This table is then read into the ciao env are the files are reszies with
        dmregrid
        @overwrite :: bool
            param decides if the table must be rewritten
        """
        

        for band in self.processed_cluster_objects[0].bandEs:
            bandname = f'{band[0] / 1000 :.1f}_{band[1] / 1000 :.1f}'
            
            t = Table(data=[self.cluster_names, self.size_og_arr], \
            names=('cluster_names', 'original_image_scale'),\
            meta={'RESIZED':self.resize_ima_to, 'BANDLOW':band[0] / 1000,\
            'BANDHIGH': band[1] / 1000, 'survey': self.survey, 'Z': self.redshift_keyword,\
            'MASS': self.mass_category})

            filename = f"{self.stack_data_location}/ToBeStackedTableInfo_{self.survey}_{self.num_objects}objs_{bandname}_{self.redshift_keyword}z_{self.mass_category}.fits"
            t.write(filename, format='fits', overwrite=overwrite)
            print(f'Saved file? {os.path.isfile(filename)}')
            
    """
    INTERMISSION
    ------------
    Process the cluster images and exposures in ciao before executing the rest of the code.

    The script that runs with ciao is named as 'ciao_stack_images.py' in the f'{self.base_dir}/code' directory
    The imaged are resized using 'dmregrid' and saved in f'{self.stack_data_location}/objects_to_stack'
    """

    def reload_and_stack(self, clobber: bool = True):
        """
        Function reloads the files generated with ciao's dmregrid and stacks them
        @data_path :: string 
            the path where the maps resized by ciao's dmcopy and dmregrid are saved
        """
        stacked_file_paths, stacked_no_mask_file_paths = np.array([]), np.array([])

        files = glob.glob(f"{self.stack_data_location}/ToBeStackedTableInfo_{self.survey}_{self.num_objects}objs_*_{self.redshift_keyword}z_{self.mass_category}.fits")
        self.logger.info(f'Found files: {files}')

        for file in files:
            this_stack_output = Table.read(file, format='fits')
            self.resize_ima_to = this_stack_output.meta['RESIZED']
            
            band = (this_stack_output.meta['BANDLOW'], this_stack_output.meta['BANDHIGH'])
            cluster_names = this_stack_output['cluster_names']
        
            bandname = f'{band[0] :.1f}_{band[1] :.1f}'
            counts_per_second_data = np.zeros((self.resize_ima_to, self.resize_ima_to))
            counts_without_mask = np.copy(counts_per_second_data)

            for i in range(len(this_stack_output)):
                inputs_folder = f"{self.stack_data_location}/{cluster_names[i]}/inputs"
                path_to_resized_ima = glob.glob(f"{inputs_folder}/ima_{bandname}_resized{self.resize_ima_to}.fits")[0]
                path_to_resized_exp =glob.glob( f"{inputs_folder}/exp_corr_{bandname}_resized{self.resize_ima_to}.fits")[0]
                path_to_resized_filled_masks = glob.glob(f"{inputs_folder}/mask_interpolated_{bandname}_resized{self.resize_ima_to}.fits")[0]
                path_to_resized_masks =glob.glob(f"{inputs_folder}/mask_resized{self.resize_ima_to}.fits")[0]
                
                # load the resized images
                image_data = fits.open(path_to_resized_ima, format='fits')[0].data
                exp_data = fits.open(path_to_resized_exp, format='fits')[0].data
                mask_data = fits.open(path_to_resized_masks, format='fits')[0].data
                filled_mask_data = fits.open(path_to_resized_filled_masks, format='fits')[0].data
                
                exp_map = exp_data.astype(np.float64)
                counts_per_second_data += (filled_mask_data)/exp_map
                counts_without_mask += image_data/exp_map
            
            # write the result
            stacked_file_path = f"{self.stack_data_location}/stacked_{self.num_objects}objs_{self.survey}_{bandname}_{self.redshift_keyword}z_{self.mass_category}.fits"
            write_fits_file(stacked_file_path, counts_per_second_data)
            
            stacked_no_mask_file_path = f"{self.stack_data_location}/stacked_no_mask_{self.num_objects}objs_{self.survey}_{bandname}_{self.redshift_keyword}z_{self.mass_category}.fits"
            write_fits_file(stacked_no_mask_file_path, counts_without_mask)
            
            stacked_file_paths = np.append(stacked_file_paths, stacked_file_path)            
            stacked_no_mask_file_paths = np.append(stacked_no_mask_file_paths, stacked_no_mask_file_path)
        self.stacked_file_paths = stacked_file_paths
        self.stacked_no_mask_file_path = stacked_no_mask_file_paths

    def plotStackedImage(self, physical_scale: float =1.5, vmin=1e-3):
        """Method to plot the stack of the cluster images 

        Parameters
        ----------
        physical_scale : float
            Physical region extracted around the cluster
        """
        for stacked_file_path in self.stacked_file_paths:
            fig, ax = plt.subplots(1,1, figsize=(6, 6))
            fig.patch.set_facecolor('white')
            
            stacked_result_data = fits.open(stacked_file_path, format='fits')[0].data
            stacked_result_data = stacked_result_data.astype(np.float64)
            stacked_result_data[np.where(stacked_result_data <= vmin)] = vmin
            
            x_len, y_len = stacked_result_data.shape
            self.logger.info(f"Shape of the resized images: {x_len}, {y_len}")
            
            df = pd.DataFrame(stacked_result_data,\
            index=np.round(np.linspace(physical_scale, -physical_scale, x_len), 1), \
            columns=np.round(np.linspace(-physical_scale, physical_scale, y_len), 1))

            cmap = mpl.cm.magma
            norm=mpl.colors.LogNorm(vmin=vmin, vmax=np.max(stacked_result_data))
            
            im = ax.imshow(df, cmap=cmap, norm=norm)
            fig.colorbar(im, ax=ax, label='counts/s')
            
            #cb1 = mpl.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm)

            ax.locator_params(axis='x', nbins=10)
            ax.locator_params(axis='y', nbins=10)
            fig.autofmt_xdate(rotation=0)

            title = f"{self.num_objects} {self.mass_category} {self.survey}\n"
            set_labels(ax, xlabel='Mpc', ylabel='Mpc', title=title, format_ticks=True, legend=False)

            if self.mass_category == 'group':
                self.mass_category = 'groups'
            fig.savefig(f'{self.image_dir}/stacked_result_{self.num_objects}_{self.survey}_resize{self.resize_ima_to}_{self.redshift_keyword}z_{self.mass_category}.png', format='png')
        return ax

    def plot_inidividual_clusters(self):
        """Function to look at the individual cluster images
        """
        energy_bands = self.processed_cluster_objects[0].bandEs

        for i, tuned_clu in enumerate(self.processed_cluster_objects):
            clu_plot_object = DoPlot(this_cluster = tuned_clu.cluster_obj, this_files = tuned_clu, \
                            data_location = self.stack_data_location)

            clu_plot_object.plot_cluster_image(do_mask_plot=True, do_exp_plot=True)