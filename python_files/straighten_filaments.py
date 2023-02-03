"""
straighten_filaments.py

This python script straightens the filaments to enable image stacking.

Script written by: Soumya Shreeram
Data created: 28th November 2022
Contact: shreeram@mpe.mpg.de

nohup python3 straighten_filaments.py 'lc_north_dis' 'SD2' '5' 'S001' '19'  > ../../data/log_files/lc_noth_dis_SD2_5_S001_straighten16_19.log &
"""

# astropy modules
import astropy.units as u
import astropy.io.fits as fits
from astropy.table import Table, Column, join
from astropy.cosmology import FlatLambdaCDM, z_at_value
from astropy.nddata import Cutout2D
from astropy import wcs

import numpy as np
import multiprocessing as mp

# system imports
import os
import sys
import glob

from reproject import reproject_interp
from reproject.mosaicking import reproject_and_coadd, find_optimal_celestial_wcs

# Load the imported file(s) that contains all the functions used in this notebooks
sys.path.append('..')
import filaments as fo

# decides wether to produce the fits file for critical points "cp", filaments "fil" or "cp_field"
keyword = "fil" 

# predefined system keywords
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

def straighten_filament_filament(filament_number):
    print(f"Hello, this is {filament_number} from process {mp.current_process()}")

    straighten_filament = fo.StraightenFilaments(data_set=data_set,
        smoothing_density_f = smoothing_density_f, 
        persistence = persistence,
        smoothing_skeleton = smoothing_skeleton,
        fil_id =filament_number)

    print(f"{filament_number=}")
    straighten_filament.append_straightened_cutouts(clobber=True)
    return


if __name__ == "__main__":
    
    this_file = fo.DisPerSEcatalog2eRass(data_set=data_set, 
        smoothing_density_f = smoothing_density_f, 
        persistence = persistence, 
        smoothing_skeleton = smoothing_skeleton,
        section_keyword = keyword)

    # selects the filaments to straighten
    selected_entires = this_file.get_selected_filament_ids()

    pool = mp.Pool(10)
    pool.map(straighten_filament_filament, selected_entires)




