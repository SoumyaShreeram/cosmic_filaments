"""
get_eRASS_skytiles_for_filaments.py

This python script does the following:
1. Obtains the sky tiles number in the eROSITA framework for every filament in the German-half of the sky. The filaments for which the skytile number is obtained also go through a selection cuts in length and orientation. 
2. Obtains the median ra, and dec for every filament
3. Creates the event file list for every filament
4. Generated the eROSITA data products (clean eventfiles, images, masks, exposures) for every filament

Script written by: Soumya Shreeram
Data created: 5th October 2022
Contact: shreeram@mpe.mpg.de
"""

# astropy modules
import astropy.units as u
import astropy.io.fits as fits
from astropy.table import Table, Column, join
from astropy.cosmology import FlatLambdaCDM, z_at_value

import numpy as np

# system imports
import os
import sys
import importlib as ib
import glob

# Load the imported file(s) that contains all the functions used in this notebooks
sys.path.append('..')
import filaments as fo

# predefined variables
if __name__ == "__main__":
    # decides wether to produce the fits file for critical points "cp", filaments "fil" or "cp_field"
    keyword = "fil" 

    # system keywords
    data_set = sys.argv[1] # 'legacy_north_dis' for Legacy MGS, 'lc_north_dis' for LOWZ+CMASS
    smoothing_density_f = sys.argv[2] # None or 'SD1'
    persistence =  float(sys.argv[3]) # 3, 4.5 or 5
    if persistence == 4.5:
        persistence = np.round(persistence, 1)
        print(f"INFO:main:persistence={persistence}")
    else:
        persistence = int(persistence)
        print(f"INFO:main:persistence={persistence}")

    smoothing_skeleton = sys.argv[4] # None or 'S001'
    
    this_file = fo.DisPerSEcatalog2eRass(data_set=data_set, 
        smoothing_density_f = smoothing_density_f, 
        persistence = persistence, 
        smoothing_skeleton = smoothing_skeleton,
        section_keyword = keyword)

    # gets the median ra, and dec for every filament
    #this_file.get_filament_medians()

    # get the cropping region around each filament (in units of Mpc)
    #this_file.get_region_around_filament()

    # gets the eRosita sky tile numbers for every filament (and filament point)
    #this_file.add_skytile_info(selection_cuts = [False, 'all'])

    # gets the event file list for every filament
    this_file.create_eventfile_list()
    
    # get data products (clean eventfiles, images, masks, exposures)
    selected_entires = this_file.get_selected_filament_ids()
    for filament_number in selected_entires:
        print(f"{filament_number=}")
        this_file.get_data_products(
            fil_id=int(filament_number))
        this_file.clean_evt()
        this_file.make_images()
        this_file.mask_clusters_point_srcs()
        #this_file.get_exposures()