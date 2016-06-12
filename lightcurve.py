import sncosmo
import numpy as np
from pytools import plotsetup
from matplotlib import pyplot as pl, ticker
from scipy import interpolate as scint
import sys
import os
import exceptions

from constants import __MJDPKNW__, __MJDPKSE__, __Z__
from constants import __MJDPREPK0NW__, __MJDPOSTPK0NW__
from constants import __MJDPREPK0SE__, __MJDPOSTPK0SE__

__THISFILE__ = sys.argv[0]
if 'ipython' in __THISFILE__:
    __THISFILE__ = __file__
__THISDIR__ = os.path.abspath(os.path.dirname(__THISFILE__))

__TEMPLATEDATAFILES__ = {
    'M31N 2008-12a (2014)':'m31n_2008-12a_2014_optical.dat',
    'U Sco':'usco.dat', 'V2487 Oph':'v2487oph.dat',
    'V394 CrA':'v394cra.dat', 'T CrB':'tcrb.dat',
    'V745 Sco':'v745sco.dat'
}


def mk_composite_template( peakmag=26.5,
        tempdatadir = 'data/RN_templates/'
        ):
    """ read in all the galactic nova templates, normalize
    and create a composite template that is normalized to the
    given peak magnitude
    :return: dict with an entry for each filter (B,V) giving the
       time array, the min, median and max magnitude values
    """
    time = np.arange(-1, 35, 0.1)
    templatedict = {'time':time}
    for band in ['B', 'V']:
        magmatrix = []
        for template in __TEMPLATEDATAFILES__.itervalues():
            if template.lower().startswith('m31'):
                continue
            tempdata = getdata(template, datadir=tempdatadir)
            interp = scint.interp1d(tempdata['t'], tempdata[band],
                                    kind='linear', bounds_error=False,
                                    fill_value=np.nan,
                                    assume_sorted=True)
            mpk = interp(0)
            m = interp(time) - mpk + peakmag
            magmatrix.append(m)
        magmax = np.nanmax(magmatrix, axis=0)
        magmin = np.nanmin(magmatrix, axis=0)
        magmed = np.nanmedian(magmatrix, axis=0)
        templatedict[band] = {'max':magmax,'min':magmin, 'med':magmed}
    return templatedict


def weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    if len(values) == 1:
        return (values[0], 1/np.sqrt(weights[0]))
    average = np.average(values, weights=weights)
    variance = np.average((values-average)**2, weights=weights)
    return (average, np.sqrt(variance))


def fit_curve_to_data(x, y, yerr, curve='gauss', t0=0.0, param0=None, verbose=False):
    """ fit a function to the given dataset
    :param dataset: a series of measurements that is presumed to be normally
       distributed, probably around a mean that is close to zero.
    :param curve: 'gauss', 'quadratic', 'cubic'
    :return: interpolator function for the best-fit curve
    """
    from scipy.optimize import curve_fit
    def gauss(t, mu, sigma):
        return np.exp(-(t-t0 - mu) ** 2 / (2 * sigma ** 2))

    def quadratic(t, a2):
        #return a1*(t-t0) + a2*(t-t0)**2
        return a2*(t-t0)**2

    def quadratic2(t, a2, a3):
        #return a1*(t-t0) + a2*(t-t0)**2
        return a2*(t-a3)**2

    def maglinear(t, a1, a2):
        return 10**(-0.4 * (a1 * (t - t0) + a2))


    def microlensing( t, t0, tE, umin):
        u = np.sqrt( umin*umin + ((t-t0)/tE)**2)
        amplification = (u*u + 2) / (u*np.sqrt(u*u+4))
        return amplification-1

    if curve=='microlensing':
        if param0==None:
            param0 = [t0, 5, 1.5]
        minparam, cov = curve_fit(microlensing, x, y, sigma=yerr, p0=param0)
        interpolator = lambda t: microlensing(t, minparam[0], minparam[1], minparam[2])
        if verbose:
            print(minparam)
        return interpolator

    if curve=='maglinear':
        if param0==None:
            param0 = [-0.1, 25]
        minparam, cov = curve_fit(maglinear, x, y, sigma=yerr, p0=param0)
        interpolator = lambda t: maglinear(t, minparam[0], minparam[1])
        if verbose:
            print(minparam)
        return interpolator

    if curve=='gauss':
        if param0==None:
            param0 = [0, 1]
        minparam, cov = curve_fit(gauss, x, y, sigma=yerr, p0=param0)
        if verbose:
            print(minparam)
        interpolator = lambda t: gauss(t, minparam[0], minparam[1])
        return interpolator

    if curve=='quadratic':
        if param0==None:
            param0 = [1, 1, 1]
        minparam, cov = curve_fit(quadratic, x, y, sigma=yerr, p0=param0)
        # interpolator = lambda t: quadratic(t, minparam[0], minparam[1])
        interpolator = lambda t: quadratic(t, minparam[0])
        if verbose:
            print(minparam)
        return interpolator

    if curve=='cubic':
        if param0==None:
            param0 = [1, 1, 1, 1]
        minparam, cov = curve_fit(cubic, x, y, sigma=yerr, p0=param0)
        interpolator = lambda t: cubic(t, minparam[0], minparam[1],
                                           minparam[2], minparam[3])
        return interpolator


def get_spock_data():
    nw = getdata('HST_FFSN_spockNW_phot.dat')
    se = getdata('HST_FFSN_spockSE_phot.dat')
    return nw, se

def getdata(datfilename, datadir="./data"):
    """  Find the given data file.
    :param datfilename:
    :param datadir:
    :return:
    """

    from os import path
    from astropy.io import ascii
    import sys

    sysdir = path.dirname(sys.argv[0])
    sysdir = path.abspath(path.expanduser(sysdir))
    filedir = path.dirname(__file__)
    filedir = path.abspath(path.expanduser(filedir))
    userdatadir = path.abspath(path.expanduser(datadir))
    sysdatadir = path.abspath(path.join(sysdir,'data/'))
    filedatadir = path.abspath(path.join(filedir,'data/'))

    for dir in [userdatadir, sysdatadir, filedatadir]:
        datfile = path.join(dir, datfilename)
        if path.exists(datfile):
            datatable = ascii.read(datfile, format='commented_header',
                                   header_start=-1, data_start=0)
            return datatable
    print("ERROR:  Could not find %s" % datfilename)
    return None



def plot_lightcurve(src='se', aperture=np.inf,
                    showtemplates=False, timeframe='rest',
                    units='mag'):
    nw, se = get_spock_data()
    if src.lower() == 'se':
        sn = se
        mjdpk = __MJDPKSE__
        fpk = 0.25
    else :
        sn = nw
        mjdpk = __MJDPKNW__
        fpk = 0.27

    if units == 'mag':
        ax = pl.gca()
        if not ax.yaxis_inverted():
            ax.invert_yaxis()

    if timeframe == 'rest':
        t = (sn['MJD'] - mjdpk) / (1 + __Z__)
    elif timeframe.startswith('obs'):
        t = sn['MJD']
    else:
        raise exceptions.RuntimeError(
            "Timeframe must be one of ['rest','obs']")

    for band, c, m in zip(['f435w', 'f606w', 'f814w',
                           'f105w', 'f125w', 'f140w', 'f160w'],
                          ['c', 'b', 'darkgreen',
                           'darkorange', 'r', 'm', 'darkred'],
                          ['o', 'o', 'o',
                           's', 's', 's', 's']):
        if units == 'flux':
            iplot = np.where((sn['FILTER'] == band.upper()) &
                             (sn['APER'] == aperture))[0]
            # flux scaling factor to be relative to ZP=25
            #f25 = 10 ** (-0.4 * (sn['ZP'][iplot] - 25))

            # flux scaling factor to get fnu in microJanskys
            fmicroJy = (1e6 * 10 ** (-0.4 * (sn['ZP'][iplot] - 8.9)))

            pl.errorbar(t[iplot], sn['FLUX'][iplot] * fmicroJy,
                        sn['FLUXERR'][iplot] * fmicroJy,
                        marker=m, color=c, ls=' ', alpha=0.5,
                        label=band.upper())
        elif units == 'mag':
            iplot = np.where((sn['FILTER'] == band.upper()) &
                             (sn['APER'] == aperture) &
                             (sn['FLUX']/sn['FLUXERR'] > 1))[0]
            pl.errorbar(t[iplot], sn['MAG'][iplot], sn['MAGERR'][iplot],
                        marker=m, color=c, ls=' ', alpha=0.5,
                        label=band.upper())

            ilim = np.where((sn['FILTER'] == band.upper()) &
                             (sn['APER'] == aperture) &
                             (sn['FLUX']/sn['FLUXERR'] < 1.5))[0]
            maglim = (-2.5 * np.log10(3 * np.abs(sn['FLUXERR'][ilim])) +
                      sn['ZP'][ilim])
            pl.errorbar(t[ilim], maglim,
                        [np.zeros(len(ilim)),np.ones(len(ilim))*0.3],
                        marker=' ', color=c, ls=' ', alpha=0.5,
                        label='__nolabel__', lolims=True)
    if showtemplates:
        tempdatadir = './data/RN_templates/'
        m31n2008a12 = __TEMPLATEDATAFILES__['M31N 2008-12a (2014)']
        tempdata = getdata(m31n2008a12, tempdatadir)
        for band, c, ls in zip(['V'],['k'],['-','--']):
            ib = np.where(tempdata['band'] == band)[0]
            mtemp = tempdata['mag'][ib]
            ttemp = tempdata['t'][ib]
            magmin = m.min()
            if timeframe.startswith('obs'):
                ttemp = (1+__Z__)*ttemp + mjdpk
            pl.plot(ttemp, mtemp - magmin + 26.5,
                    ls=ls, color='0.4', marker=' ')
        composite = mk_composite_template(peakmag=26.5,
                                          tempdatadir=tempdatadir)
        for band, c, ls in zip(['V'],['k','g'],['-','--']):
            ttemp = composite['time']
            min = composite[band]['min']
            max = composite[band]['max']
            if timeframe.startswith('obs'):
                ttemp = (1+__Z__)*ttemp + mjdpk
            pl.fill_between(ttemp, min, max,
                            color=c, label='__nolabel__', alpha=0.15)
        if src=='nw':
            if timeframe.startswith('obs'):
                xtext = lambda x: x*(1+__Z__) + mjdpk
            else:
                xtext = lambda x: x
            pl.text(xtext(2.5), 26.7, 'M31N 2008a-12', color='k',
                    ha='left', va='center')
            pl.plot([xtext(0.85), xtext(2.3)], [27.35,26.7],
                    ls='-', color='k', lw=0.5)
            pl.text(xtext(4), 27.1, 'MW RNe', color='0.5',
                    ha='left', va='center')
            pl.plot([xtext(2.3), xtext(3.9)], [27.4,27.05],
                    ls='-', color='0.3', lw=0.5)

def fit_function_to_lightcurve(fitfunction=True):
    foo = """
            itofit = np.where((sn['FILTER'] == band.upper()) &
                              (sn['APER'] == aperture) &
                              (sn['MJD'] > mjdpk-40) &
                              (sn['MJD'] < mjdpk))[0]
            if len(itofit) == 0 : continue
            ff = 10 ** (-0.4 * (sn['ZP'][itofit] - 25))


            fluxtofit = sn['FLUX'][itofit]*ff
            fluxerrtofit = sn['FLUXERR'][itofit]*ff
            mjdtofit = sn['MJD'][itofit]

            # pl.plot( mjdtofit, fluxtofit, marker='x', color='k', ls='-')
            weights = 1/(fluxerrtofit**2)

            ## Fit an interpolating polynomial to these data
            #interpolator = fit_curve_to_data(mjdtofit, fluxtofit, fluxerrtofit,
            #                                 'gauss', [mjdpk, 5])

            print(band.upper())

            interpolator = fit_curve_to_data(mjdtofit, fluxtofit, fluxerrtofit,
                                             curve=fitfunction, t0=mjdpk,
                                             param0=[mjdpk, 5.0, 1.5],
                                             verbose=True)

            # fit an interpolating spline to these data
            mjdtoplot = np.arange(mjdtofit.min()-10,mjdpk,0.1)
            pl.plot(mjdtoplot, interpolator(mjdtoplot), ls='-', color=c, lw=0.7)

            #pl.errorbar(mjdtofit, fluxtofit, fluxerrtofit,
            #            marker='x', ls='--', color=c, lw=0.7)
"""


def mklcfig_single(event='se', presfig=True, units='mag',
                   axlabels=True, showlegend=False, **kwargs):
    """ Make a single light curve figure in the current AXES instance.
    :param presfig: size the figure and lines for presentations
    :return:
    """
    fig = pl.gcf()
    axobs = fig.gca()

    axobs.xaxis.set_major_locator(ticker.MultipleLocator(10))
    # axobs.xaxis.set_major_locator( ticker.MaxNLocator(2))
    axobs.xaxis.set_minor_locator(ticker.MultipleLocator(2))
    axobs.ticklabel_format(useOffset=False, style='plain')

    if event.lower() in ['se', 2]:
        tpk = __MJDPKSE__
        axobs.set_xlim(tpk - 12, tpk + 18)
        axobs.text(0.12, 0.92, '2 (SE)', fontsize='x-large',
                transform=axobs.transAxes, ha='left', va='top')
    elif event.lower() in ['nw', 1]:
        tpk = __MJDPKNW__
        axobs.set_xlim(tpk - 12, tpk + 18)
        axobs.text(0.12, 0.92, '1 (NW)', fontsize='x-large',
                transform=axobs.transAxes, ha='left', va='top')
    else:
        raise RuntimeError("Event name %s not recognized."
                           "Use one of ['se', 'nw', 1, 2]")

    axrest = axobs.twiny()
    plot_lightcurve(event.lower(), units=units, timeframe='rest', **kwargs)
    # axobs.axvspan(-tpkerr,tpkerr,color='0.5', alpha=0.5)
    axrest.set_xlim((axobs.get_xlim()[0] - tpk) / 2.0054,
                    (axobs.get_xlim()[1] - tpk) / 2.0054)
    axrest.xaxis.set_major_locator(ticker.MultipleLocator(5))
    axrest.xaxis.set_minor_locator(ticker.MultipleLocator(1))

    if units == 'flux':
        axrest.set_ylim(-0.019, 0.098)
        axrest.axhline(0.0, ls='--', lw=0.8, color='0.5')
    elif units == 'mag':
        axrest.set_ylim(29.9, 26.1)

    axobs.xaxis.set_ticks_position('top')
    axobs.xaxis.set_label_position('top')
    axobs.xaxis.set_label_coords(-0.03, 1.02)
    axobs.xaxis.set_tick_params(pad=1)

    axrest.xaxis.set_ticks_position('bottom')
    axrest.xaxis.set_label_position('bottom')
    if presfig:
        axrest.xaxis.set_label_coords(-0.12, -0.03)
    else:
        axrest.xaxis.set_label_coords(-0.03, -0.03)

    if axlabels:
        axobs.set_xlabel('MJD :')
        if units == 'flux':
            axobs.set_ylabel('f$_{\\nu}$~[$\\mu$Jy]', labelpad=8)
        elif units == 'mag':
            axobs.set_ylabel('Apparent Magnitude (AB)', labelpad=8)
        axrest.set_xlabel('t$_{\\rm rest}$ :')

    if showlegend:
        axrest.legend(loc='upper right', ncol=2, numpoints=1,
                     columnspacing=0.5,
                     borderpad=0.4, borderaxespad=0.5,
                     labelspacing=0.3, markerscale=0.8,
                     fontsize='small', framealpha=1,
                     frameon=False,
                     handletextpad=0.35, handlelength=0.2)
    # axrest.axhline(0.0, ls='--', color='0.5', lw=0.75)
    # axrest.axvline(0.0, ls='--', color='0.5', lw=0.75)
    return axrest, axobs


def mklcfig_double(presfig=False, **kwargs):
    """ Make a double light curve figure.
    :param presfig: size the figure and lines for presentations
    :return:
    """
    if presfig:
        plotsetup.presfig(figsize=[10, 6])
    else:
        plotsetup.fullpaperfig([8, 3.0])

    pl.clf()
    fig = pl.gcf()
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2, sharey=ax1)

    for event, ax in zip(
            ['nw', 'se'], [ax1, ax2]):
        fig.sca(ax)
        axrest, axobs = mklcfig_single(event, presfig=presfig,
                                       axlabels=(event == 'nw'),
                                       showlegend=(not presfig and
                                                   event == 'se'),
                                       **kwargs)
        if event=='se':
            pl.setp(axobs.get_xticklabels()[0:2], visible=False)
        elif event=='nw':
            pl.setp(axobs.get_xticklabels()[-2:], visible=False)

    ax2.yaxis.set_ticks_position('right')
    ax2.yaxis.set_ticks_position('both')

    if presfig:
        fig.subplots_adjust(left=0.15, bottom=0.1, right=0.9, top=0.88,
                            wspace=0.05)
    else:
        fig.subplots_adjust(left=0.1, bottom=0.1, right=0.94, top=0.92,
                            wspace=0.03)

    #ax1.invert_yaxis()
    #ax2.invert_yaxis()

    pl.draw()




def mk_prediction_fig(presfig=False):
    """  make a figure showing predictions and excluded regions
    in time delay space for the two possible arrival order
    sequences
    :return:
    """
    from astropy.io import ascii
    from matplotlib import pyplot as pl, cm
    import numpy as np
    from matplotlib import ticker
    from pytools import plotsetup

    tpkNW = 56670
    tpkSE = 56908


    if presfig:
        plotsetup.presfig( figsize=[12,6])
    else :
        plotsetup.halfpaperfig( figsize=[8,4])
    pl.clf()
    fig = pl.gcf()

    ax1 = fig.add_subplot(2, 1, 1)
    plot_lightcurve(src='nw', units='flux', timeframe='observer')

    ax2 = fig.add_subplot(2, 1, 2, sharex=ax1, sharey=ax1)
    plot_lightcurve(src='se', units='flux', timeframe='observer')

    # read in the time delay data
    timedelaydatfile = os.path.join(__THISDIR__,
                                    'data/time_delay_predictions.txt')
    dtdat = ascii.read(timedelaydatfile,
                       format='commented_header', header_start=-1,
                       data_start=0 )
    for row in dtdat:
        if row['model'] == 'observed':
            continue
        tnw = tpkSE - row['dt12']
        tnwerr = row['dt12err']
        ax1.axvspan(tnw-tnwerr, tnw+tnwerr, color='darkred', alpha=0.3)

        tse = tpkNW + row['dt12']
        tseerr = row['dt12err']
        ax2.axvspan(tse-tseerr, tse+tseerr, color='darkblue', alpha=0.3)
        # print('%s %.1f +- %.1f' % (row['model'], row['dt12'], row['dt12err']))


    ax1.axhline(0, color='0.5', lw=0.6, ls='--')
    ax2.axhline(0, color='0.5', lw=0.6, ls='--')
    ax1.set_xlim(56575, 57048)
    ax1.set_ylim(-0.03, 0.099)

    ax1.set_ylabel('Flux [$\\mu$Jy]', labelpad=-1)
    ax2.set_ylabel('Flux [$\\mu$Jy]', labelpad=-1)
    ax2.set_xlabel('MJD :')
    ax2.xaxis.set_label_coords(0.0, -0.06)


    ymin, ymax = ax1.get_ylim()
    yevent = ymin + (ymax - ymin) * 0.6
    ytextHigh = ymin + (ymax - ymin) * 0.9
    ytextMid = ymin + (ymax - ymin) * 0.7

    ax2.text(56726, ytextHigh,
             '\\noindent Predicted appearance\\\\of Spock-1 event at\\\\'
             'the SE position',
             ha='left', va='top', color='darkblue')
    ax2.annotate('\\noindent Spock-2\\\\ event',
                 xy=(56890, yevent), xytext=(56870, ytextMid),
                 ha='right', va='top', color='darkred',
                 arrowprops=dict(color='darkred', width=0.5, headwidth=3.5,
                             shrink=0))

    ax1.text(56850, ytextHigh,
             '\\raggedleft \\noindent Predicted appearance\n'
             'of Spock-2 event at \n the NW position',
             ha='right', va='top', color='darkred')
    ax1.annotate('\\noindent Spock-1\\\\ event',
                 xy=(56680, yevent), xytext=(56700, ytextMid),
                 ha='left', va='top', color='darkblue',
                 arrowprops=dict(color='darkblue', width=0.5, headwidth=3.5,
                                 shrink=0))

    ax2.xaxis.set_major_locator(ticker.MultipleLocator(50))
    ax2.xaxis.set_minor_locator(ticker.MultipleLocator(25))

    pl.setp(ax1.get_xticklabels(),visible=False)
    ax2.xaxis.set_ticks_position('bottom')
    ax2.xaxis.set_label_position('bottom')

    ax1.text(56590, ytextHigh, '\\noindent NW position\\\\(image 11.2)',
             va='top', ha='left', # transform=ax1.transAxes,
             fontsize='large', weight='heavy')
    ax2.text(57038, ytextHigh, '\\noindent  SE position\\\\(image 11.1)',
             va='top', ha='right', # transform=ax2.transAxes,
             fontsize='large', weight='heavy')
    # ax1.text(56885, ytextHigh, 'J',   color='k', ha='center', va='top')
    # ax1.text(56911, ytextHigh, 'O',   color='k', ha='center', va='top')
    # ax1.text(56932, ytextHigh, 'J,Z,W', color='k', ha='center', va='top')
    # ax1.text(56955, ytextHigh, 'D',   color='k', ha='center', va='top')
    # #ax2.text(56692, ytextHigh, 'J',   color='k', ha='center', va='top')
    # ax2.text(56649, ytextHigh, 'W,Z,J', color='k', ha='center', va='top')
    # ax2.text(56620, ytextHigh, 'D',   color='k', ha='center', va='top')
    # ax2.text(56666, ytextHigh, 'O',   color='k', ha='center', va='top')

    fig.subplots_adjust(left=0.09, right=0.98, top=0.97, bottom=0.1, hspace=0)


def microlensing( t, t0, tE, umin):
    u = np.sqrt( umin*umin + ((t-t0)/tE)**2)
    amplification = (u*u + 2) / (u*np.sqrt(u*u+4))
    return amplification-1
