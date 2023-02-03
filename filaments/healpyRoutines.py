"""
healpyRoutines.py

This python file has the common functions used in conjunction with healpy

Script written by: Soumya Shreeram
Date created: 11th May 2022
Contact: shreeram@mpe.mpg.de
"""
# astropy modules
import astropy.units as u
import astropy.io.fits as fits

from astropy.table import Table, Column, join
from astropy.coordinates import SkyCoord
from astropy.cosmology import FlatLambdaCDM, z_at_value
from gdpyc import GasMap

import numpy as np

# system imports
import os
import sys
import importlib as ib
import glob
import gzip
import copy

# plotting imports
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D

import seaborn as sns
import pandas as pd

import healpy as hp
from healpy.newvisufunc import projview, newprojplot

def ra_dec_2_theta_phi(ra,dec):
    """
    Function to convert the ra and dec into theta and phi

    Parameters
    ----------
    ra :: arr, float
        The array with the ra of the points
    dec :: arr, float
        The array with the dec of the points
    """
    c = SkyCoord(ra=ra, dec=dec, frame='icrs', unit=u.deg)
    phi, theta = c.ra.wrap_at(180*u.deg).radian, 0.5 * np.pi - c.dec.radian
    return theta, phi

def produce_healpy_map(NSIDE, ra, dec, return_pixel_idx=False, fil_val=None, nest=True):
    """
    Function to generate the filament map as accepted by healpy
    
    """
    
    NPIX = hp.nside2npix(NSIDE)
    filament_map = np.zeros(NPIX)

    theta, phi = ra_dec_2_theta_phi(ra, dec)
    pixel_idx = hp.pixelfunc.ang2pix(NSIDE, theta, phi, nest=nest)
    
    if fil_val is None:
        fil_vale = 1

    filament_map[pixel_idx] = fil_val
    
    if return_pixel_idx:
        return filament_map, pixel_idx
    else:
        return filament_map

def get_how_many_neighbours(NSIDE, ra, dec, return_pixel_idx=False):
    filament_map, pixel_idx = produce_filament_map(NSIDE, ra, dec, return_pixel_idx=True)

    neighbours = hp.pixelfunc.get_all_neighbours(NSIDE, pixel_idx, nest=True)
    vals_neighbours = np.zeros(neighbours.shape)
    vals_neighbours = filament_map[neighbours]
    if return_pixel_idx:
        return vals_neighbours, neighbours
    else:
        return vals_neighbours

def healpy_mollview(healpy_map=None, cmap=plt.cm.gray, title='',
    norm=None, vmin=1e-2, vmax=None, nest=True, rot = None, NSIDE=512, ra=None, dec=None, projection_type='mollweide'):
    """
    Function to plot the healpy map in mollweide projection 

    Parameters
    ----------
    healpy_map :: 
    cmap :: string
        matplotlib colormap
    title :: string
        Title of the plot 
    norm :: matplotlib property
        healpy options inclue 'log' and 'hist'
    """
    if healpy_map is None:
        if ra is None or dec is None:
            raise NameError('input ra, dec, and NSIDE as no map provided.')
        healpy_map = produce_healpy_map(NSIDE, ra, dec)

    if vmax is None:
        vmax=np.max(healpy_map)
        
    if norm is 'LogNorm':
        vmin = np.exp(1)
        norm=mpl.colors.LogNorm(vmin=vmin, vmax=vmax)
        print("INFO:healpyRoutines.py: ", norm)

    cmap = copy.copy(mpl.cm.get_cmap(cmap))
    ax = projview(healpy_map, projection_type=projection_type,
        nest=nest, xlabel='R.A.', ylabel='Dec',
        title=title,
        cmap=cmap, 
        norm=norm,
        rot=rot,
        coord='C',
        min=vmin,
        max=vmax,
        cbar=False, 
        ytick_label_color='k',
        hold=True,
        override_plot_properties={"xlabel_color":"w", 
        'figure_width': 10, 'figure_size_ratio': 0.63},
        graticule=True,
        graticule_labels=True,
        fontsize={"xlabel": 18, "ylabel":18}
        )
    return 

def healpy_cartesian(healpy_map=None, cmap=plt.cm.gray, title='',
    norm=None, vmin=1e-2, vmax=None, nest=True, rot = None, NSIDE=512, ra=None, dec=None, projection_type='mollweide'):
    """
    Function to plot the healpy map in mollweide projection 

    Parameters
    ----------
    healpy_map :: 
    cmap :: string
        matplotlib colormap
    title :: string
        Title of the plot 
    norm :: matplotlib property
        healpy options inclue 'log' and 'hist'
    """
    if healpy_map is None:
        if ra is None or dec is None:
            raise NameError('input ra, dec, and NSIDE as no map provided.')
        healpy_map = produce_healpy_map(NSIDE, ra, dec)

    if vmax is None:
        vmax=np.max(healpy_map)
        
    if norm is 'LogNorm':
        vmin = np.exp(1)
        norm=mpl.colors.LogNorm(vmin=vmin, vmax=vmax)
        print("INFO:healpyRoutines.py: ", norm)

    cmap = copy.copy(mpl.cm.get_cmap(cmap))
    ax = projview(healpy_map, projection_type=projection_type,
        nest=nest, xlabel='R.A.', ylabel='Dec',
        title=title,
        cmap=cmap, 
        norm=norm,
        rot=rot,
        coord='C',
        min=vmin,
        max=vmax,
        cbar=False, 
        ytick_label_color='k',
        hold=True,
        override_plot_properties={"xlabel_color":"w", 
        'figure_width': 10, 'figure_size_ratio': 0.63},
        graticule=True,
        graticule_labels=True,
        fontsize={"xlabel": 18, "ylabel":18}
        )
    return

def get_milky_way(plot_map=False, fil_val=4, NSIDE: int=256):
    """Function to produce the map that plots the milky way equator in Celestial coordinates
    """
    # milky way in galactic coordinates
    c = SkyCoord(l=np.linspace(0, 360, 10000), b=np.zeros(10000), unit='degree', frame='galactic') 
    ra_milky_way, dec_milky_way = c.fk5.ra, c.fk5.dec
     
    milky_way_map = produce_healpy_map(NSIDE, ra_milky_way, dec_milky_way, fil_val=fil_val)
    if plot_map:
        healpy_mollview(milky_way_map, vmax=1.2, title=f'center at ra={0}')
        custom_xtick_labels=[ra_c_LOWZ-60, ra_c_LOWZ-30, ra_c_LOWZ, ra_c_LOWZ+30, ra_c_LOWZ+60]
    return milky_way_map

def twoDarray2map(skeleton, NSIDE=256):
    """
    Function to reconstruct the 1-D healpy map array from the 2-D projected array
    NSIDE :: int
        The healpix nside parameter, must be a power of 2, less than 2**30
    """
    NPIX = hp.pixelfunc.nside2npix(NSIDE)
    filament_skeleton_map = np.zeros(NPIX)
    theta, phi = hp.pixelfunc.pix2ang(NSIDE, np.arange(NPIX) )
    
    proj_fil_skeleton_map = np.zeros(skeleton.shape)

    # changing the skeleton map into a 1/0 array
    proj_fil_skeleton_map[skeleton] = 1

    proj = hp.projector.CartesianProj(coord='C',
        xsize=proj_fil_skeleton_map.shape[1], ysize=proj_fil_skeleton_map.shape[0])

    x, y = proj.ang2xy(theta=theta, phi=phi, lonlat=False)
    i, j = proj.xy2ij(x=x, y=y)
 
    pixel_idx = hp.pixelfunc.ang2pix(NSIDE, theta, phi, lonlat=False)
    pixel_idx_nest = hp.pixelfunc.ring2nest(NSIDE, pixel_idx)

    filament_skeleton_map[pixel_idx] = proj_fil_skeleton_map[i, j]
    filament_skeleton_map = hp.pixelfunc.reorder(filament_skeleton_map, inp='RING', out='NEST', r2n=True)
    return filament_skeleton_map

def get_nh(ra,dec,res='high', return_pixel_idx=False, nh_cut = 0.1, NSIDE_in=512):
    """
    Function to get the Nh value for the points where the ra, and dec is given
    """
    coords=SkyCoord(ra, dec, unit = 'deg')
    if res=='high':
        nh_vals = GasMap.nh(coords, nhmap = 'HI4PI', hires = True).value/1e22
        NSIDE_out = 1024
    if res=='low':
        nh_vals = GasMap.nh(coords, nhmap = 'HI4PI', hires = False).value/1e22
        NSIDE_out = 512

    NPIX = hp.nside2npix(NSIDE_out)
    nh_map = np.zeros(NPIX)

    theta, phi = ra_dec_2_theta_phi(ra, dec)
    pixel_idx = hp.pixelfunc.ang2pix(NSIDE_out, theta, phi, nest=True)

    foreground_dominated = np.where(nh_vals>nh_cut)
    nh_vals[foreground_dominated] = 0
    nh_map[pixel_idx] = nh_vals

    if NSIDE_out != NSIDE_in:
        nh_map = hp.pixelfunc.ud_grade(nh_map, NSIDE_in, order_in='NESTED', order_out='NESTED', dtype=float)
    return nh_map