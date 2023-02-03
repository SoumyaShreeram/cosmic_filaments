"""
StraightenFilaments.py

This python file owns the class for straightening the filaments 

Script written by: Soumya Shreeram
Date created: 23rd October 2022
Contact: shreeram@mpe.mpg.de
"""
import numpy as np

from astropy.io import fits
from astropy.table import QTable, Table, Column, join
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy import wcs
from astropy.cosmology import z_at_value, Planck18, FlatLambdaCDM
from astropy.visualization import simple_norm
from astropy.nddata.utils import Cutout2D

from sympy import Point, Segment, Line
import sympy as sp

import os
import subprocess
import glob
import logging
from scipy.ndimage import gaussian_filter

from reproject import reproject_interp, reproject_adaptive
from reproject.mosaicking import reproject_and_coadd, find_optimal_celestial_wcs

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import healpy as hp
from .DisPerSEcatalog import DisPerSEcatalog
from .DisPerSEcatalog2eRass import DisPerSEcatalog2eRass
from .utils import set_labels, custom_colormap, set_as_white

class StraightenFilaments(DisPerSEcatalog2eRass):
    """
    File to straighten the filaments from the erosita image files
    """
    def __init__(self, data_set: str = 'lc_north_dis', smoothing_density_f: str = "None", persistence: float = 3, smoothing_skeleton: str = "None", section_keyword: str = 'fil', fil_id: int = None) -> None:
        """
        Initialized when an object of StraightenFilaments is called

        Parameters
        ----------
        # copy common parameters from DisPerSEcatalog2eRass
        data_set :: str
            the dataset on which the skeleton has been detected 
            ('legacy_north_dis' for Legacy MGS, 'lc_north_dis' for LOWZ+CMASS)
        smoothing_density_f :: str
            the smoothing of the density field (None or 'SD1' for 1 smoothing cycle ...)
        persistence :: int
            the persistence threshold ('3' or '5')
        smoothing_skeleton :: str
            the smoothing of the skeleton (None or 'S001' for 1 smoothing cycle ...)
        filament_name :: str
            same as fil_id, if filament name is specified, the concerned filament is analysed
            Note that the filament name must be in the following format:
            "filid"
        fil_id :: int
            the filament number used for the analysis
        

        """
        logging.basicConfig(level = logging.INFO)
        self.logger = logging.getLogger(type(self).__name__)

        # initialized the interited class parameters
        super().__init__(data_set=data_set, smoothing_density_f=smoothing_density_f, persistence=persistence, smoothing_skeleton=smoothing_skeleton)

        t = Table.read(f'{self.reformatted_filename}.fit')

        if fil_id is None:
            fil_ids_selected = self.get_selected_filament_ids()
        
            # select the first selected filament 
            fil_id = fil_ids_selected[0]
            self.logger.info(f" choosing {fil_id=}")

        # get the data products for this filament (usually already generated)
        self.get_data_products(fil_id=int(fil_id))


    def get_mosiac_image(self, rotate=False):
        """Function to get the mosaic of the sub-filament images

        Parameters
        ----------
        rotate :: bool (default: False)
            bool decides wether to also rotate the images with respect to the straight axis
        """
        self.rotate = rotate

        # get the names and the centers of the concerned image
        coord = np.loadtxt(f"{self.inputs_folder}/image_centers.txt")

        images_to_plot = glob.glob(f"{self.inputs_folder}/ima_0.2_2.3_*.fits.gz")
        hdus = [fits.open(filename)[0] for filename in images_to_plot]
        wcs_out, shape_out = find_optimal_celestial_wcs(hdus)
        wcs_out.wcs.ctype = ['RA---SIN', 'DEC--SIN']
        
        # gets the new mosaic centers or the filament points
        x_c, y_c = self.get_new_filament_centres(wcs_out)


        if rotate:
            self.logger.info(" First obtaining the WCS of the filament points without rotation")
            _, _, wcs_img_arr = self.get_image_properties()
            self.update_fil_headers(x_c, y_c, wcs_img_arr)
            
            hdus_rot = [fits.open(filename) for filename in self.out_ima_rot_files]
            wcs_rot_out, shape_rot_out = find_optimal_celestial_wcs(hdus_rot)
        
        
            whole_fil_array, whole_fil_footprint = reproject_and_coadd(hdus_rot,
                                                   wcs_rot_out, shape_out=shape_rot_out,
                                                   reproject_function=reproject_interp)
            
        else:
            whole_fil_array, whole_fil_footprint = reproject_and_coadd(hdus,
                                                   wcs_out, shape_out=shape_out,
                                                   reproject_function=reproject_interp)
            wcs_rot_out = f"Dict not generated; change keyword `{rotate=}`"

        whole_fil_array[np.isnan(whole_fil_array)] = 0
        whole_fil_footprint[np.isnan(whole_fil_footprint)] = 0
        return whole_fil_array, whole_fil_footprint, wcs_out, wcs_rot_out

    def get_pixel_numbers(self, wcs_img, ra, dec):
        """Method to perform simple conversions between (ra, dec) -> (x, y) 
        
        Parameters
        ----------
        wcs_img :: dict 
            the WCS header for the concerned image
        ra, dec :: (float, float)
            the ra and dec of the point that needs to be converted into pixel units

        Returns
        -------
        x, y :: (float, float)
            the pixel hosting the inputted ra and dec
        """
        x, y = wcs_img.all_world2pix([[ra,dec]],0)[0][0], wcs_img.all_world2pix([[ra,dec]],0)[0][1]
        return x, y

    def rotation_matrix(self, scale: float = 1, rot_angle = 0.):
        """A simple rotate matrix for rotating 2D images
        Parameters
        ----------

        """
        self.logger.info(f"{rot_angle=:.5f} rad")
        pc11 = scale*sp.cos(rot_angle)
        pc12 = scale*sp.sin(rot_angle)
        pc21 = -scale*sp.sin(rot_angle)
        pc22 = scale*sp.cos(rot_angle)
        return [[pc11, pc12], [pc21, pc22]]

    def transform_vector(self, x, y, rot_angle, x0, y0, scale: float = 1):
        x_new = (x-x0)*scale*sp.cos(rot_angle) + (y-y0)*scale*sp.sin(rot_angle)
        y_new = - (x-x0)*scale*sp.sin(rot_angle) + (y-y0)*scale*sp.cos(rot_angle)
        return x_new, y_new
    
    def define_PCtransformation_matrix_sympy(self, cdelt1, cdelt2, rot_angle = 0):
        """
        Linear transformation matrix. See R. J. Hanisch (1988) for further details
        URL to website: https://lweb.cfa.harvard.edu/~jzhao/SMA-FITS-CASA/docs/wcs88.pdf
        Last checked on 14th Nov 2022

        Parameters
        ----------
        rot_angle :: float (unit: radians)
            the angle by which the image must be rotated
        cdelt1, cdelt2 :: 
        """

        cd11 = cdelt1*sp.cos(rot_angle)
        cd12 = sp.Abs(cdelt2)*sp.sign(cdelt1)*sp.sin(rot_angle)
        cd21 = -sp.Abs(cdelt1)*sp.sign(cdelt2)*sp.sin(rot_angle)
        cd22 = cdelt2*sp.cos(rot_angle)
        return [[cd11, cd12], [cd21, cd22]]

    def get_image_properties(self):
        """Function to get the pixel coordinates of the filament points in the WCS of the pixel
        
        Returns
        -------
        x_c_fil, y_c_fil :: (float, float)
            the filament center in the WCS of the filament sub-image
        wsc_img_arr :: 1darray of type dict 
            the WSC header for each of the filament sub-images

        """
        # get the names and the centers of the concerned image
        images_to_plot = glob.glob(f"{self.inputs_folder}/ima_0.2_2.3_*.fits.gz")
        coord = np.loadtxt(f"{self.inputs_folder}/image_centers.txt")

        x_c_fil, y_c_fil = np.array([]), np.array([])
        wcs_img_arr = np.array([], dtype=object)
        for i in range(len(coord[0])):
            hdu_img = fits.open(images_to_plot[i])
            header_img = hdu_img[0].header
            data_img = hdu_img[0].data
            
            header_img['RADESYSa'] = 'ICRS'
            wcs_img = wcs.WCS(naxis=2)
            wcs_img.wcs.crpix = data_img.shape[0]/2, data_img.shape[1]/2
            wcs_img.wcs.cdelt = -1, 1
            wcs_img.wcs.pc = self.rotation_matrix()
            wcs_img_arr = np.append(wcs_img_arr, wcs_img)
            
            ra, dec = coord[0, i], coord[1, i]
            x, y = self.get_pixel_numbers(wcs_img, ra, dec)
            x_c_fil= np.append(x_c_fil, x)
            y_c_fil= np.append(y_c_fil, y)
        return x_c_fil, y_c_fil, wcs_img_arr

    def get_new_filament_centres(self, wcs_out, rotate=False):
        """Function to get the new filament center with the inputted WCS dict

        Parameters
        ----------
        wcs_out :: dict
            the optimal WCS header for the mosiac image
        """
        if not rotate:
            coord = np.loadtxt(f"{self.inputs_folder}/image_centers.txt")
            median_x_c, median_y_c = self.get_pixel_numbers(wcs_out, self.median_ra, self.median_dec)
            
            ra_arr, dec_arr = coord[0, :], coord[1, :]
            self.median_x_c = median_x_c
            self.median_y_c = median_y_c
        else:
            rotated_centers = np.loadtxt(self.rotated_centers_filename)
            ra_arr = rotated_centers[np.arange(0,len(rotated_centers),step=2)]
            dec_arr = rotated_centers[np.arange(1,len(rotated_centers)+1,step=2)]

        x_c, y_c = np.array([]),  np.array([]) 
        for ra, dec in zip(ra_arr, dec_arr ):
            x, y = wcs_out.all_world2pix([[ra, dec]],0)[0][0], wcs_out.all_world2pix([[ra, dec]],0)[0][1]
            x_c, y_c = np.append(x_c, x), np.append(y_c, y)
        return x_c, y_c

    def get_perpendicular(self, segment = None):
        """Function to get the point and line perpendicular to the input segment
        
        Parameters
        ----------
        segment :: object of class Segment (from sympy package)
            the input segment whose perpendicular properties are found out
        
        Returns
        -------
        point_perpendicular :: object of class Point (from sympy package)
            the point on the line perpendicular to the input segment
        perpendicular_line :: object of class Line (from sympy package)
            the line perpendicular to the input segment
        """
        if segment is None:
            raise Warning()
        perpendicular_line = segment.perpendicular_bisector()
        r = perpendicular_line.random_point(seed=42)  # seed value is optional
        point_perpendicular = r.n(3)
        return point_perpendicular, perpendicular_line

    def get_rotated_WCS(self, wcs_img_arr, x_c, y_c, dy: float = 100):
        """Function calculates the angle by which each image is rotated.
        The information is stored in the WSC array in the PCi_j input
    
        Parameters
        ---------
        wcs_img_arr :: 1darray of type dict
            the array holds the WSC information of all the filament sub-images
        x_c, y_c :: (1darray, 1darray) of type float   
            the pixel coordinates in the frame of the mosaic image
        dy :: float (default value set to 100)
            the increment to the y-axis (in image/pixel frame) to get the "straight" filament segment
        """
        wcs_rotimg_arr = wcs_img_arr
        rot_angles = np.zeros(len(x_c))
        filament_points = self.generate_filament_points()
        for i in range(len(wcs_img_arr)):    
            point1, point2 = Point(x_c[i], y_c[i]), Point(filament_points[i][0], filament_points[i][1])
            distance = sp.N(point1.distance(point2))
            self.logger.info(f"distance between fil segment start point and mid-point: {distance:.2f}")
            
            # if it is not the first point that is overlapping
            if (distance > 0) and (i != 0):
                fil_segment = Segment(point1, point2)
                # get the perpendicular point to the filament
                point_perpen_fil, line_perpen_fil = self.get_perpendicular(fil_segment)

                # gettting the straight filament 
                median_point, median_point2 = Point(self.median_x_c, self.median_y_c), Point(self.median_x_c, self.median_y_c+dy)
                straight_fil = Segment(median_point, median_point2)

                # get the perpendicular point to the straightened filament
                point_perpen_str_fil, line_perpen_str_fil = self.get_perpendicular(straight_fil)

                angle_between_lines = line_perpen_str_fil.smallest_angle_between(line_perpen_fil)
                angle_in_radian = sp.N(angle_between_lines)
                self.logger.info(f"angle of image rotation = {angle_in_radian:.3f}")
                rot_angles[i] = angle_in_radian
                
                # add rotation to the cutout header
                wsc_of_img = wcs_img_arr[i]
                wsc_of_img.wcs.pc = self.rotation_matrix(rot_angle=angle_in_radian)
                wcs_rotimg_arr[i] = wsc_of_img
            else:
                rot_angles[i] = rot_angles[i-1]
                wcs_rotimg_arr[i] = wcs_rotimg_arr[-1]
            
        rot_angle_mean = np.mean(rot_angles)
        self.rot_angles = rot_angles
        return wcs_rotimg_arr

    def update_fil_headers(self, x_c, y_c, wcs_img_arr):
        """[DEPRICATED] Function to get the pixel coordinates of the filament points in the WCS of the pixel
        Parameters
        ---------
        wcs_img_arr :: 1darray of type dict
            the array holds the WSC information of all the filament sub-images
        x_c, y_c :: (1darray, 1darray) of type float   
            the pixel coordinates in the frame of the mosaic image
        """
        # get the names and the centers of the concerned image
        images_to_plot = glob.glob(f"{self.inputs_folder}/ima_0.2_2.3_*.fits.gz")
        coord = np.loadtxt(f"{self.inputs_folder}/image_centers.txt")

        wcs_rot_arr = self.get_rotated_WCS(wcs_img_arr, x_c, y_c)
        
        out_ima_rot_files = np.array([], dtype=object)
        coord_rot_arr = np.array([])
        for i in range(len(coord[0])):
            hdu_img = fits.open(images_to_plot[i])
            header_img = hdu_img[0].header
            data_img = fits.open(images_to_plot[i])[0].data
            
            # change the header entries 
            wcs_img = wcs_rot_arr[i]
            header_img['PC1_1'] = wcs_img.wcs.pc[0][0]
            header_img['PC1_2'] = wcs_img.wcs.pc[0][1]
            header_img['PC2_1'] = wcs_img.wcs.pc[1][0]
            header_img['PC2_2'] = wcs_img.wcs.pc[1][1]
            
            # get the rotated filament points
            x_cen_rot_pix, y_cen_rot_pix = data_img.shape[0]//2, data_img.shape[1]//2
            ra_cen_rot = wcs_img.pixel_to_world_values([[x_cen_rot_pix, y_cen_rot_pix]])[0][0]
            dec_cen_rot = wcs_img.pixel_to_world_values([[x_cen_rot_pix, y_cen_rot_pix]])[0][1]
            coord_rot_arr = np.append(coord_rot_arr, [ra_cen_rot, dec_cen_rot])

            # change the input data
            rotated_data, footprint = reproject_interp((data_img, wcs_img), header_img, shape_out=data_img.shape)

            # save info in a new Primary Hdu
            hdu_img_new = fits.PrimaryHDU(rotated_data,header=header_img)
            out_ima_rot_filename = f"{self.inputs_folder}/ima_rot_0.2_2.3_{i}.fits.gz"
            out_ima_rot_files = np.append(out_ima_rot_files, out_ima_rot_filename)
            hdu_img_new.writeto(out_ima_rot_filename, overwrite=True)     
        
        # method outputs
        self.rotated_centers_filename = f"{self.rescaled_folder}/rotated_image_centers.txt"
        np.savetxt(self.rotated_centers_filename, coord_rot_arr)
        self.out_ima_rot_files = out_ima_rot_files
        

    def plot_whole_filament(self, wcs_out, whole_fil_array, whole_fil_footprint, wcs_rot_out = None, save_fig=True, rotate=None, plt_colorbar = False):
        """Method to plot the mosaic filament

        Parameters
        ----------
        wcs_out :: dict
            the optimal WCS header for the mosiac image (with/without any rotation applied)
        whole_fil_array :: 2D numpy array
            the output of `reproject_and_coadd` that gives a mosaic output
        whole_fil_footprint :: 2D numpy array (same shape as whole_fil_array)
            the second output of `reproject_and_coadd` and this gives the footprint of the mosaic output
        save_fig :: bool (default: True)
            saves the generated images to the self.rescaled_outputs directory
        """
        if rotate is None:
            rotate = self.rotate
        if rotate:
            x_c, y_c = self.get_new_filament_centres(wcs_rot_out, rotate=rotate)
            set_wcs = wcs_rot_out
        else:
            median_x_c, median_y_c = self.median_x_c, self.median_y_c
            x_c, y_c = self.get_new_filament_centres(wcs_out)
            set_wcs = wcs_out

        fig = plt.figure(figsize=(15, 10))
        fig.set_facecolor('black')
        ax1 = plt.subplot(1,2,1, projection=set_wcs)
        ax1 = plt.subplot(1, 2, 1)
        vmax = np.max(whole_fil_array[~np.isnan(whole_fil_array)])
        vmin = np.min(whole_fil_array[~np.isnan(whole_fil_array)])
        self.logger.info(f"{vmin=}, {vmax=}")

        # plot the mosaic
        norm = simple_norm(whole_fil_array, 'log', min_cut=1e-2, max_cut=.75)
        smoothed_data = gaussian_filter(whole_fil_array, sigma=3)
        im1 = ax1.imshow(smoothed_data, origin="lower", norm=norm, cmap=custom_colormap())
        
        # plot the filament points
        ax1.plot(x_c, y_c, ls='', marker='*', color='#d66860',ms=20, zorder=2, mec='w')

        if not rotate:
            # plot the filament median
            ax1.plot(median_x_c, median_y_c, 'ws', zorder=3, mec='k')

        if plt_colorbar:
            fig.subplots_adjust(right=0.85)
            cbar_ax = fig.add_axes([0.88, 0.15, 0.04, 0.7])
            fig.colorbar(im1, cax=cbar_ax)
        set_labels(ax1, r"R.A. [hh$^{mm}$]", 'Dec [deg]', 'Combimed filament image')
        set_as_white(ax1)
        
        ax2 = plt.subplot(1,2,1, projection=set_wcs)
        ax2 = plt.subplot(1, 2, 2)
        norm_fp = simple_norm(whole_fil_footprint, 'log', min_cut=1e-2, max_cut=10)
        smoothed_fp_data = gaussian_filter(whole_fil_footprint, sigma=3)
        im2 = ax2.imshow(smoothed_fp_data, origin="lower", norm=norm_fp, cmap=custom_colormap())
        set_labels(ax2, "X [pixels]", 'Y [pixels]', 'Footprint of filament images')
        set_as_white(ax2)

        if save_fig:
            plt.savefig(f"{self.rescaled_folder}/mosaic_fil{self.fil_id}_{self.rotate=}.png", format='png', bbox_inches='tight', facecolor='w')

        return ax1, ax2

    def get_crop_sizes_WCS(self, wcs_out, overlap=True, rotate=True):
        """Function to get the cut out sizes for every rotateed filament sub-
        """
        # need to get the rotated wsc of the images
        _, _, wcs_img_arr = self.get_image_properties()
        x_c, y_c = self.get_new_filament_centres(wcs_out, rotate=False)
        if rotate:
            wcs_rotimg_arr = self.get_rotated_WCS(wcs_img_arr, x_c, y_c)
            wcs_arr = wcs_rotimg_arr
        else:
            wcs_arr = wcs_img_arr
        
        size_arr = np.zeros(len(wcs_arr))
        coord_fil_points = self.generate_filament_points()
        # need to know what these transform to in pixel units
        for i, wcs_img in enumerate(wcs_arr):
            ra_filseg_start, dec_filseg_start = coord_fil_points[i]
            ra_filseg_end, dec_filseg_end = coord_fil_points[i+1]
            
            hdu_rot_i = fits.open(self.out_ima_rot_files[i])[0]
            data_rot_img_i = hdu_rot_i.data
            header_rot_i = hdu_rot_i.header
            
            x_start, y_start = self.get_pixel_numbers(wcs_img, ra_filseg_start, dec_filseg_start)
                
            x_end, y_end = self.get_pixel_numbers(wcs_img, ra_filseg_end, dec_filseg_end)
            size_arr[i] = x_end-x_start

        if overlap:
            idx = np.where(size_arr <= 0)[0]
            if (len(wcs_arr)-1) not in idx:
                size_arr[idx] = size_arr[idx+1]
            else:
                idx_new = idx[:-1]
                size_arr[idx_new] = size_arr[idx_new+1]
                size_arr[-1] = size_arr[-2]
        return size_arr

    def get_CD_matrix(self, wcs_img_arr, x_c, y_c, dy: float = 100):
        """Function calculates the angle by which each image is rotated.
        The information is stored in the WSC array in the PCi_j input
    
        Parameters
        ---------
        wcs_img_arr :: 1darray of type dict
            the array holds the WSC information of all the filament sub-images
        x_c, y_c :: (1darray, 1darray) of type float   
            the pixel coordinates in the frame of the mosaic image
        dy :: float (default value set to 100)
            the increment to the y-axis (in image/pixel frame) to get the "straight" filament segment
        """
        rot_angles = np.zeros(len(x_c))
        cd_matrix_arr = []
        filament_points = self.generate_filament_points()
        for i in range(len(wcs_img_arr)):    
            point1, point2 = Point(x_c[i], y_c[i]), Point(filament_points[i][0], filament_points[i][1])
            distance = sp.N(point1.distance(point2))
            self.logger.info(f"distance between fil segment start point and mid-point: {distance:.2f}")
            
            # if it is not the first point that is overlapping
            if (distance > 0) and (i != 0):
                fil_segment = Segment(point1, point2)
                # get the perpendicular point to the filament
                point_perpen_fil, line_perpen_fil = self.get_perpendicular(fil_segment)

                # gettting the straight filament 
                median_point, median_point2 = Point(self.median_x_c, self.median_y_c), Point(self.median_x_c, self.median_y_c+dy)
                straight_fil = Segment(median_point, median_point2)

                # get the perpendicular point to the straightened filament
                point_perpen_str_fil, line_perpen_str_fil = self.get_perpendicular(straight_fil)

                angle_between_lines = line_perpen_str_fil.smallest_angle_between(line_perpen_fil)
                angle_in_radian = sp.N(angle_between_lines)
                self.logger.info(f"angle of image rotation = {angle_in_radian:.3f}")
                rot_angles[i] = angle_in_radian
            else:
                rot_angles[i] = rot_angles[i-1]
        if rot_angles[0] == 0:
            rot_angles[0] = rot_angles[1]

        for angle in rot_angles:
            # add rotation to the cutout header
            cd_matrix = self.rotation_matrix(rot_angle=angle)
            cd_matrix_arr.append(cd_matrix)
        self.rot_angles = rot_angles
        return cd_matrix_arr, rot_angles


    def get_crop_sizes(self):
        """Function to get the pixel coordinates of the filament segment endpoints in the original unrotated image
        
        Returns
        -------
        x_c_fil, y_c_fil :: (float, float)
            the filament segment ends in the WCS of the filament sub-image
        """
        # get the names and the centers of the concerned image
        images_to_plot = glob.glob(f"{self.inputs_folder}/ima_0.2_2.3_*.fits.gz")
        filament_points = self.generate_filament_points()
        coord = np.loadtxt(f"{self.inputs_folder}/image_centers.txt")
        ra_fil_midpts, dec_fil_midpts = coord[0], coord[1]

        x_c_fil, y_c_fil = [], []
        for i in range(len(images_to_plot)):
            ra_start, dec_start =  filament_points[i]
            ra_end, dec_end = filament_points[i+1]
            
            hdu_img = fits.open(images_to_plot[i])
            header_img = hdu_img[0].header
            data_img = hdu_img[0].data
            wcs_img = wcs.WCS(header_img)
            wcs_img.wcs.crpix = data_img.shape[0]//2, data_img.shape[1]//2
            wcs_img.wcs.crval = ra_fil_midpts[i], dec_fil_midpts[i]
            
            x_start, y_start = self.get_pixel_numbers(wcs_img, ra_start, dec_start)
                
            x_end, y_end = self.get_pixel_numbers(wcs_img, ra_end, dec_end)
            
            x_c_fil.append([x_start, x_end])
            y_c_fil.append([y_start, y_end])
        return np.array(x_c_fil), np.array(y_c_fil)

    def get_rotated_crop_sizes(self, x_fil_ends, y_fil_ends, size):
        """Function to get the new filament segment end points after the rotation is applied to the images

        Parameters
        ----------
        @x_fil_ends ::
            ndarray of shape (N, 2) where the start and end x-coordinates of the filament segments ends are stored
        @y_fil_ends ::
            ndarray of shape (N, 2) where the start and end y-coordinates of the filament segments ends are stored
        @size :: 
            the size of the filament sub-image 
        """
        x_fil_new, y_fil_new = [], [] 

        # center of the filament sub-image
        x0, y0 = size//2, size//2
        for i, angle in enumerate(self.rot_angles):
            x_start, x_end = x_fil_ends[i]
            y_start, y_end = y_fil_ends[i]
            x_start_new, y_start_new = self.transform_vector(x_start, y_start, angle, x0, y0)
            x_end_new, y_end_new = self.transform_vector(x_end, y_end, angle, x0, y0)
            
            x_fil_new.append([x_start_new+x0, x_end_new+x0])
            y_fil_new.append([y_start_new+y0, y_end_new+y0])
        return np.array(x_fil_new), np.array(y_fil_new)

    def get_rotated_filament_pixels(self):
        """Function to get the filament segment ends, both the original and rotated, in pixel units

        Outputs
        -------
        self.rotated_centers_pix_filename : str
            the name of the file where the rotated filament segment ends are stored
        self.centers_pix_filename : str
            the name of the file where the original filament segment ends are stored
        """
        # get the output WCS for converting filament points into pixel units
        images_to_plot = glob.glob(f"{self.inputs_folder}/ima_0.2_2.3_*.fits.gz")
        hdus = [fits.open(filename)[0] for filename in images_to_plot]
        wcs_out, shape_out = find_optimal_celestial_wcs(hdus)
        wcs_out.wcs.ctype = ['RA---SIN', 'DEC--SIN']

        # get the centers of the filament segments in pixel units
        x_c, y_c = self.get_new_filament_centres(wcs_out)
        _, _, wcs_img_arr = self.get_image_properties()

        # the rotation matrix and the rotation angles per image
        cd_matrix_arr, rot_angles = self.get_CD_matrix(wcs_img_arr, x_c, y_c)

        # the filament ends required for cropping the images
        x_fil_ends, y_fil_ends = self.get_crop_sizes()

        # the size of the filament sub-image
        size = fits.open(images_to_plot[0])[0].data.shape[0]
        
        # the filament ends after applying the rotation to the image
        x_fil_new, y_fil_new = self.get_rotated_crop_sizes(x_fil_ends, y_fil_ends, size)

        self.rotated_centers_pix_filename = f"{self.rescaled_folder}/rotated_image_centers_pixels.txt"
        np.savetxt(self.rotated_centers_pix_filename, [x_fil_new[:, 0], x_fil_new[:, 1], y_fil_new[:, 0], y_fil_new[:, 1]])

        self.centers_pix_filename = f"{self.rescaled_folder}/image_centers_pixels.txt"
        np.savetxt(self.centers_pix_filename, [x_fil_ends[:, 0], x_fil_ends[:, 1], y_fil_ends[:, 0], y_fil_ends[:, 1]])
        return cd_matrix_arr, rot_angles

    def get_straighted_images(self, filenames, masks, sub_img_no, cd_matrix):
        """Function to get the straightened filament cut outs in a given filament sub-image
        """
        i = sub_img_no 
        # open the file whose rotated cutout is obtained
        hdu_test = fits.open(filenames[i])
        mask_data = fits.open(masks[i])[0].data
        
        input_data = (hdu_test[0].data)*mask_data
        cdelt = wcs.utils.proj_plane_pixel_scales(wcs.WCS(hdu_test[0].header))


        # define a simple input WCS
        input_wcs = wcs.WCS(naxis=2)
        input_wcs.wcs.crpix = input_data.shape[0]/2, input_data.shape[1]/2
        input_wcs.wcs.cdelt = -cdelt[0], cdelt[1]
        input_header = input_wcs.to_header()
        
        # defube an output WCS with rotation, defined in the CD matix
        output_wcs = wcs.WCS(naxis=2)
        output_wcs.wcs.crpix = input_data.shape[0]/2, input_data.shape[1]/2
        output_wcs.wcs.cdelt = input_wcs.wcs.cdelt
        output_wcs.wcs.pc = cd_matrix

        # rotation of the image
        result_gaussian, _ = reproject_adaptive((input_data, input_wcs), 
                                        output_wcs, 
                                        shape_out=input_data.shape)

        fil_ends_og = np.loadtxt(self.centers_pix_filename)
        x_start, x_end, y_start, y_end = (fil_ends_og.T)[i]
        
        fil_ends_rot = np.loadtxt(self.rotated_centers_pix_filename)
        x_start_rot, x_end_rot, y_start_rot, y_end_rot = (fil_ends_rot.T)[i]
        
        # get the cutouts of the original and rotated image
        cutout_width = x_end - x_start
        cutout_img = Cutout2D(input_data, (input_data.shape[0]/2, input_data.shape[1]/2), 
                          (input_data.shape[0]-20, cutout_width))

        rotated_cutout_width = x_end_rot - x_start_rot
        cutout_rot_img = Cutout2D(result_gaussian, (input_data.shape[0]/2, input_data.shape[1]/2), 
                          (result_gaussian.shape[0]-20, rotated_cutout_width))

        return fil_ends_og, fil_ends_rot, input_data, result_gaussian, cutout_img, cutout_rot_img, output_wcs

    def append_straightened_cutouts(self, clobber=False):
        """Function to get the straightened filament cutouts put together
        """
        sub_image_filenames = glob.glob(f"{self.inputs_folder}/ima_0.2_2.3_*.fits.gz")
        sub_mask_filenames = glob.glob(f"{self.inputs_folder}/mask_*.fits.gz")
        
        cd_matrix_arr, _ = self.get_rotated_filament_pixels()
        
        straightened_fil_name = f"{self.rescaled_folder}/straightened_0.2_2.3_{len(sub_image_filenames)-1}.fits.gz"
        straightened_fil_ends =  f"{self.rescaled_folder}/straightened_fil_centers_pixels.txt"
        if not os.path.isfile(straightened_fil_name) or not os.path.isfile(straightened_fil_ends) or clobber:
            straighted_fil = np.array([])
            new_x_ends, new_y_ends = [], []
            for i in range(len(sub_image_filenames)):
                out = self.get_straighted_images(sub_image_filenames, sub_mask_filenames, i, cd_matrix_arr[i])
                _, _, _, _, _, cutout_rot_img, output_wcs = out
                if i == 0:
                    straighted_fil = cutout_rot_img.data.T

                    size_x, size_y = straighted_fil.shape[0]/2, straighted_fil.shape[1]/2
                    new_x_ends.append(size_x), new_y_ends.append(size_y)
                    self.logger.info(f"adding filament segment {i+1}/{len(sub_image_filenames)}")
                else:
                    size_x, size_y = straighted_fil.shape[0]+(cutout_rot_img.data.T.shape[0]/2), straighted_fil.shape[1]/2
                    straighted_fil = np.append(straighted_fil, cutout_rot_img.data.T, axis=0)
                    self.logger.info(f"adding filament segment {i+1}/{len(sub_image_filenames)}")
                    
                    new_x_ends.append(size_x), new_y_ends.append(size_y)

            hdu_straight = fits.PrimaryHDU(straighted_fil.T, header=output_wcs.to_header())
            hdu_straight.writeto(straightened_fil_name, overwrite=True) 

            np.savetxt(straightened_fil_ends, np.array([new_x_ends, new_y_ends])) 
        
        self.straightened_fil_name = straightened_fil_name 
        self.straightened_xyfil_ends = straightened_fil_ends
        return 


