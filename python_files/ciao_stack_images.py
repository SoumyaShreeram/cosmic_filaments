"""
ciao_stack_images.py

Script used to execute the stacking of cluster images using Chandra's ciao software
"""
import numpy as np
import logging
import os
import sys
import subprocess

# astropy modules
import astropy.units as u
import astropy.io.fits as fits
from astropy.table import Table

# input variables
survey = 'eRASS1'
num_objects = 30


DATA_DIR = "/data53s/shreeram/Cluster_image_stacking/data/stacking_test"

def copy_regrid_images(data_location: str = DATA_DIR, clobber_dmcopy: str = "yes", \
 clobber_dmregrid: str = "yes", survey: str = 'eRASS1', num_objects: int = 20, \
 bandname: str = '0.5_2.0', resized_to: int = 60):
    """
    Function to copy and regrid the cluster images before stacking them


    dmregrid parameters
    -------------------
        Input image 
        output file name, 
        Binning specification, 
        CCW rotation angle in degrees about rotation center
        x, y offset
        x, y coordinate of rotation center
        Number of points in pixel (0='exact' algorithm) (0:999)
    """

    this_stack_output = Table.read(f"{DATA_DIR}/ToBeStackedTableInfo_{survey}_{num_objects}objs_{bandname}.fits", format='fits')
    
    # log the outputs
    logger = logging.getLogger(__name__)
    logging.basicConfig(level = logging.INFO)

    for i in range(len(this_stack_output)):
        scale_factor = this_stack_output['scale_factors'][i]
        size_og = int(this_stack_output['original_image_scale'][i])
        cluster_name = this_stack_output['cluster_names'][i]
        logger.info(f"----------- \n{cluster_name}\n -----------")
        
        # paths from    
        ima_path = f"{DATA_DIR}/{cluster_name}/inputs/ima_{bandname}.fits"
        exp_path_corr =f"{DATA_DIR}/{cluster_name}/inputs/exp_corr_{bandname}.fits"    
        mask_path = f"{DATA_DIR}/{cluster_name}/inputs/mask.fits"
        filled_mask_path = f"{DATA_DIR}/{cluster_name}/inputs/mask_interpolated_{bandname}.fits"
        
        # new file creation and copying to desired location
        resized_ima  = f"{DATA_DIR}/{cluster_name}/inputs/ima_{bandname}_resized{resized_to}"
        resized_exp  = f"{DATA_DIR}/{cluster_name}/inputs/exp_corr_{bandname}_resized{resized_to}"
        resized_masks = f"{DATA_DIR}/{cluster_name}/inputs/mask_resized{resized_to}"
        resized_filled_masks  = f"{DATA_DIR}/{cluster_name}/inputs/mask_interpolated_{bandname}_resized{resized_to}"
        
        file_type_arr = np.array(['image', 'corrected exposure', 'mask', 'filled mask'])
        input_paths = np.array([ima_path, exp_path_corr, mask_path, filled_mask_path])
        output_paths = np.array([resized_ima, resized_exp, resized_masks, resized_filled_masks])

        check_if_infiles_exist = np.any([os.path.isfile(in_files) for in_files in input_paths])
        logger.info(f"All input files exists? {check_if_infiles_exist}") 

        for in_path, out_path, file_type in zip(input_paths, output_paths, file_type_arr):
            out_file = f"{out_path}_dmcopy.fits"
            
            dmcopy = ["dmcopy",
            f"{in_path}", 
            out_file,
            f"clobber={clobber_dmcopy}"
            ]

            subprocess.run(dmcopy, capture_output=True)
            logger.info(dmcopy)
            logger.info(f"Copied {file_type}? {os.path.isfile(out_file)}")
            
            
            # resize images
            resized_out_file = f"{out_path}.fits"
            dmregrid=["dmregrid",
            f"{out_file}", 
            resized_out_file,
            f"1:{size_og}:{scale_factor:.2f},1:{size_og}:{scale_factor:.2f}",
            "0",
            "0",
            "0",
            "0",
            "0",
            "0",
            f"clobber={clobber_dmregrid}"]

            subprocess.run(dmregrid, capture_output=True)
            logger.info(dmregrid)
            logger.info(f"Resized {file_type}? {os.path.isfile(resized_out_file)}")
        
        

if __name__ == "__main__":
    copy_regrid_images(survey=survey, num_objects=num_objects)

