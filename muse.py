import numpy as np
from pytools import plotsetup, spec
from matplotlib import pyplot as pl, ticker
from scipy import interpolate as scint
from astropy.table import Table
import sys
import os
import exceptions
import glob

from constants import __Z__, __THISDIR__


class Spec1D(object):
    def __init__(self, specfilename, varfilename):
        self.specdata = None
        self.vardata = None
        self.use_normalized_flux = False
        self.normfactor = 1

        for filename, yunit in zip([specfilename, varfilename],
                                   ['flux','variance']):
            fileinsrcdir = os.path.join(__THISDIR__, filename)
            spockdatadir = os.path.join(__THISDIR__, 'data/MUSE/')
            fileindatadir = os.path.join(spockdatadir, filename)
            if os.path.isfile(filename):
                fullpath = filename
            elif os.path.isfile(fileindatadir):
                fullpath = fileindatadir
            elif os.path.isfile(fileinsrcdir):
                fullpath = fileinsrcdir
            else:
                raise exceptions.RuntimeError(
                    "I can't find the file %s" % filename)

            datatable = Table.read(fullpath, format='ascii.basic',
                                   names=['wavelength', 'value'])
            if yunit=='flux':
                self.wavelength = datatable['wavelength']
                self.fluxraw = datatable['value']
            else:
                self.variance = datatable['value']
                self.fluxerrraw = np.sqrt(datatable['value'])

    def normalize_to_OII_peak_flux(self, peakwidth=1.5):
        """ define a normalization factor that will rescale the flux so
        that the OII line at 3729 Angstroms (restframe)
        reaches a peak value of 1
        """
        woii = 3729 * (1 + __Z__)
        ioii = np.where(np.abs(self.wavelength - woii) < peakwidth)[0]
        peakflux = np.median(self.flux[ioii])
        self.normfactor = 1.0 / peakflux
        self.use_normalized_flux = True
        return


    @property
    def flux(self):
        if self.use_normalized_flux:
            return self.fluxraw * self.normfactor
        else:
            return self.fluxraw

    @property
    def fluxerr(self):
        if self.use_normalized_flux:
            return self.fluxerrraw * self.normfactor
        else:
            return self.fluxerrraw

    @property
    def flux_upper(self):
        if self.use_normalized_flux:
            return (self.fluxraw + self.fluxerrraw) * self.normfactor
        else:
            return self.fluxraw + self.fluxerrraw

    @property
    def flux_lower(self):
        if self.use_normalized_flux:
            return (self.fluxraw - self.fluxerrraw) * self.normfactor
        else:
            return self.fluxraw - self.fluxerrraw


    def plot_points(self, ax=None, **kwargs):
        if ax is None:
            ax = pl.gca()
        plotargs = {'color': 'darkcyan', 'ls': ' ',
                    'marker': 'o'}
        plotargs.update(**kwargs)
        ax.errorbar(self.wavelength, self.flux, self.fluxerr, **plotargs)
        return


    def plot_line(self, ax=None, **kwargs):
        if ax is None:
            ax = pl.gca()
        plotargs = {'color': 'darkcyan', 'ls': '-',
                    'marker': ' '}
        plotargs.update(**kwargs)
        ax.plot(self.wavelength, self.flux, **plotargs)
        return


    def plot_errband(self, ax=None, **kwargs):
        if ax is None:
            ax = pl.gca()
        plotargs = {'color': 'k', 'alpha': 0.3}
        plotargs.update(**kwargs)

        ax.fill_between(self.wavelength, self.flux_lower, self.flux_upper,
                        **plotargs)
        return


    def plot_smoothed(self, smoothingwindow=5, ax=None, **kwargs):
        if ax is None:
            ax = pl.gca()
        plotargs = {'marker':' ', 'ls':'-', 'color':'darkorange',
                    'drawstyle': 'steps'}
        plotargs.update(**kwargs)

        fsmooth = spec.savitzky_golay(self.flux, window_size=smoothingwindow)
        ax.plot(self.wavelength, fsmooth, **plotargs)
        return


    def plot_binned(self, binwidth=5, ax=None, **kwargs):
        if ax is None:
            ax = pl.gca()
        plotargs = {'marker':'o', 'ls':' ', 'color':'darkcyan',
                    'drawstyle':'steps'}
        plotargs.update(**kwargs)

        wbinned, dwbinned, fbinned, dfbinned = spec.binspecdat(
            self.wavelength, self.flux, self.fluxerr, binwidth=binwidth)
        ax.errorbar(wbinned, fbinned, dfbinned, dwbinned, **plotargs)
        return


def get_speclist():
    speclist = []
    for i in range(1, 11):
        specfile = 'spec_%i.txt' % i
        varifile = 'vari_%i.txt' % i
        spec1d = Spec1D(specfile, varifile)
        speclist.append(spec1d)

    for ispock in [1,2]:
        specfile = 'spec_spock%i.txt' % ispock
        varifile = 'vari_spock%i.txt' % ispock
        spec1d = Spec1D(specfile, varifile)
        speclist.append(spec1d)
    return speclist


def plot_OII_sequence(speclist=None, showspockpositions=False, normalize=True):
    if speclist is None:
        speclist = get_speclist()
    plotsetup.fullpaperfig(figsize=[8, 3.5])
    fig = pl.gcf()
    fig.clf()
    ax1 = fig.add_subplot(1, 10, 1)

    specspock1 = speclist[-2]
    specspock2 = speclist[-1]
    if normalize:
        specspock1.normalize_to_OII_peak_flux()
        specspock2.normalize_to_OII_peak_flux()

    for i in range(1, 11):
        spec1d = speclist[i - 1]
        if normalize:
            spec1d.normalize_to_OII_peak_flux()
        if i > 1:
            ax = fig.add_subplot(1, 10, i, sharex=ax1, sharey=ax1)
        else:
            ax = ax1
        # spec1d.plot_errband(ax=ax)
        spec1d.plot_line(ax=ax, color='k', drawstyle='steps', lw=0.7)
        if showspockpositions:
            specspock1.plot_line(ax=ax, color='darkcyan',
                                 drawstyle='steps', lw=0.7)
            specspock2.plot_line(ax=ax, color='darkorange',
                                 drawstyle='steps', lw=0.7)
    polish_specplot()
    return


def polish_specplot():
    fig = pl.gcf()
    ax1 = fig.axes[0]
    ax1.set_xlim(7452, 7498)
    ax1.set_ylim(-0.05, 1.2)
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax1.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(30))
    ax1.xaxis.set_minor_locator(ticker.MultipleLocator(5))

    fig.subplots_adjust(left=0.08, bottom=0.15, right=0.98, top=0.97,
                        hspace=0, wspace=0)
    for i in range(len(fig.axes)):
        if i == 0:
            continue
        ax = fig.axes[i]
        pl.setp(ax.get_yticklabels(), visible=False)
    axmid = fig.axes[5]
    axmid.set_xlabel('Observed Wavelength ($\AA$)')
    ax1.set_ylabel('Normalized Flux')
    ax1.set_yticklabels([])

    pl.draw()
    return


def plot_OII_image(fitsfile='data/MUSE/MACSJ0416_stack_7475_7482.fits'):
    """ plot the OII image from the data cube
    """
    from astropy.io import fits
    from matplotlib import cm
    fitsfile = os.path.join(__THISDIR__, fitsfile)
    fitsimage = fits.open(fitsfile)
    pl.clf()
    ax = pl.gca()
    hostslice1 = fitsimage[0].data[135:215, 215:315]
    ax.imshow(hostslice1, cmap=cm.Greys, aspect='equal',
              interpolation='nearest', alpha=None,
              vmin=-15, vmax=30, origin='lower', extent=None)



def plot_OII_linefitdata(datafile='data/MUSE/muse_hostgal_OII_line_fits.txt'):
    """ make a plot showing results from Italo Balestra's simple gaussian
    fits to the OII lines
    :return:
    """
    from astropy.io import ascii
    datafile = os.path.join(__THISDIR__, datafile)
    datatable = ascii.read(datafile, format='commented_header',
                           header_start=-1, data_start=0)
    pl.clf()
    ax = pl.gca()
    ax.plot(datatable['d_spock2'],
            datatable['fluxratio3729to3726'] /
            datatable['fluxratio3729to3726'][-1],
            ls=' ', marker='o', color='r')
    ax.plot(datatable['d_spock2'],
            datatable['flux3726']/datatable['flux3726'][-1],
            ls=' ', marker='d', color='b')
    ax.plot(datatable['d_spock2'],
            datatable['flux3729']/datatable['flux3729'][-1],
            ls=' ', marker='d', color='g')
    ax.set_xlabel('distance from spock-2')
    ax.set_ylabel('OII flux ratio')


