"""
util.py
This file contains miscillaneous functions used in the other files

"""
import numpy as np
import sys

import astropy.units as u
from astropy.coordinates import SkyCoord
import astropy.io.fits as fits
from astropy.cosmology import FlatLambdaCDM, z_at_value
from astropy import wcs

import logging
import matplotlib.pyplot as plt
import matplotlib as mpl

def aitoffProjection(sky_ra, sky_dec):
    c = SkyCoord(ra=sky_ra, dec=sky_dec, frame='icrs')
    ra, dec = c.ra.wrap_at(180*u.deg).radian, c.dec.radian
    return ra, dec


def custom_colormap():
    cm = np.loadtxt('/data53s/shreeram/Filament_stacking/data/sls.lut')
    N = len(cm)
    vals = np.ones((N, 4))
    vals[:, 0] = cm[:, 0]
    vals[:, 1] = cm[:, 1]
    vals[:, 2] = cm[:, 2]
    newcm = mpl.colors.ListedColormap(vals)
    return newcm

def set_labels(ax, xlabel, ylabel, title='', xlim='default', ylim='default',\
 legend=False, format_ticks=False, set_as_white=True, log_scale=[False, 'xy']):
    """
    Function defining plot properties
    @param ax :: axes to be held
    @param xlabel, ylabel :: labels of the x-y axis
    @param title :: title of the plot
    @param xlim, ylim :: x-y limits for the axis
    """
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    if xlim != 'default':
        ax.set_xlim(xlim)
    
    if ylim != 'default':
        ax.set_ylim(ylim)
    
    if legend:
        l = ax.legend(loc='best',  fontsize=14, frameon=False)
        for legend_handle in l.legendHandles:
            legend_handle._legmarker.set_markersize(12)
        
    if format_ticks:    
        ax.tick_params(axis='x', which='major',  length=6, width=1)
        ax.tick_params(axis='y', which='major',  length=6, width=1)

    if set_as_white: 
        color='w'
    else:
        color='k'
    ax.set_title(title, fontsize=18, color=color)
    ax.grid(False)

    if log_scale[0]:
        if log_scale[1] == 'x':
            ax.set_xscale('log')
        if log_scale[1] == 'y':
            ax.set_yscale('log')
        if log_scale[1] == 'xy':
            ax.set_xscale('log')
            ax.set_yscale('log')
    return

def set_as_white(ax):
    ax.yaxis.label.set_color('w')
    ax.xaxis.label.set_color('w')

    ax.tick_params(axis='x', colors='w', which='both')
    ax.tick_params(axis='y', colors='w', which='both')
    return ax

def cen_pos(ra: float = None, dec: float = None, evt_file: str = None):
        """
        Method is used to define the center position of the clusters 
        Parameters
        ----------
        @ra, dec :: float
            the corrected ra and dec of the filament point
        @evt_file :: str
            parameter to decide which event file to use to set the center position

        Outputs
        -------
        xval, yval :: float
            the x and y values in the detector coordinates
        """
        if ra is None or dec is None or evt_file is None:
            self.logger.info("No ra or dec given or evt_file given")
        
        # get x, y pixel column
        f = fits.open(evt_file)
        hdu = f['events']
        xcol, ycol = hdu.data.columns['X'], hdu.data.columns['Y']
        w = wcs.WCS(naxis=2)
        w.wcs.crpix = [xcol.coord_ref_point, ycol.coord_ref_point]
        w.wcs.cdelt = [xcol.coord_inc, ycol.coord_inc]
        w.wcs.crval = [xcol.coord_ref_value, ycol.coord_ref_value]
        w.wcs.ctype = [xcol.coord_type, ycol.coord_type]
        xval, yval = w.wcs_world2pix(ra, dec, 1)  # count from 1, not 0 
        return xval, yval

def astropy_dist(ra1, dec1, ra2, dec2):
    coord1 = SkyCoord(ra1 * u.deg, dec1 * u.deg, frame='icrs')
    coord2 = SkyCoord(ra2 * u.deg, dec2 * u.deg, frame='icrs')
    dist = coord1.separation(coord2).to(u.deg).value
    return dist


def circle(X, Y):
    x, y = np.meshgrid(X, Y)
    rho = np.sqrt(x * x + y * y)
    return rho