"""
reformatting_Malavasi_catalog2fits.py

This python script reformats the ascii files in the Malavasi et al. 2020 catalog into fits format for easier access

Script written by: Soumya Shreeram
Data created: 24th August 2022
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

if __name__ == "__main__":
	"""
	Input parameters
	================
	"""
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

	"""
	Defining the object and running associated methods
	==================================================
	"""
	this_file = fo.DisPerSEcatalog(data_set=data_set, 
		smoothing_density_f = smoothing_density_f, 
		persistence = persistence, 
		smoothing_skeleton = smoothing_skeleton,
		section_keyword = keyword)

	# coverts the ascii file into fits format with columns
	this_file.convert_ascii2fits(clobber=True)

	# writes another table for the (x, y, z) coordinated converted to (ra, dec, redshift)
	this_file.get_ra_dec_z()

	# calculated the filament lengths
	this_file.get_filament_lengths()

	# calculates the nH values of the filaments, to check if affected by foreground absorption
	this_file.select_low_nh_filaments(nh_key='conservative')

	# gets the angle of the filaments with respect to the line of sight
	this_file.get_orientation_filaments()

	# get the elongation or quantity to measure the "straightness" of the filament
	this_file.get_elongation_filaments()

	# checks in the filament is the German-half of the eROSITA sky ('DE' or 'RU')
	this_file.check_if_filament_in_eRASSde()