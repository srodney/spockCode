from __future__ import print_function
from constants import (__MJDPKNW__, __MJDPKSE__, __Z__, __H0__ , __OM__,
                       __MJDPREPK0NW__, __MJDPOSTPK0NW__,
                       __MJDPREPK0SE__, __MJDPOSTPK0SE__,
                       __VEGARESTBANDNAME__, __ABRESTBANDNAME__)

from . import lightcurve
from scipy import interpolate as scint
from scipy import optimize as scopt
import numpy as np
from matplotlib import pyplot as pl, ticker, rcParams, cm
import matplotlib.transforms as mtransforms
from matplotlib.patches import (Ellipse, FancyBboxPatch, Rectangle,
                                FancyArrowPatch)
from pytools import plotsetup, cosmo
from astropy.io import ascii
import sncosmo
import sys
import os
import exceptions

# constant combining the speed of light with the standard luminosity factor
# to convert from V band AB mags to Luminosity in erg/s
__cL0__ = 0.13027455 # x10^40 erg s-1 Angstrom   (the 10^40 is handled below)
MVfromLogL = lambda logL: -2.5*(logL - np.log10(__cL0__ / 5500) - 40)
logLfromMV = lambda MV: np.log10(__cL0__ / 5500) + 40 - (0.4 * MV)

__THISFILE__ = sys.argv[0]
if 'ipython' in __THISFILE__:
    __THISFILE__ = __file__
__THISDIR__ = os.path.abspath(os.path.dirname(__THISFILE__))



def draw_bbox(ax, bb):
    # boxstyle=square with pad=0, i.e. bbox itself.
    p_bbox = FancyBboxPatch((bb.xmin, bb.ymin),
                            abs(bb.width), abs(bb.height),
                            boxstyle="square,pad=0.",
                            ec="k", fc="none", zorder=10.,
                            )
    ax.add_patch(p_bbox)


def linear_fit_light_curves(linfitbands=['f435w', 'f814w', 'f125w', 'f160w'],
                            figsize='tall', declinetimemetric='t2'):
    """ Fit the rise and decline of the light curve with
     a straight line in mag vs time.  Measure the rise time and decline time
     for each event in the given bands, for a range of assumed dates of peak
     brightness.
    """
    assert figsize in ['wide','tall']
    if figsize=='wide':
        fig = plotsetup.fullpaperfig(figsize=[8,3])
        fig.subplots_adjust(left=0.1, bottom=0.1, hspace=0.15,
                            right=0.95, top=0.95)
    else:
        fig = plotsetup.halfpaperfig(figsize=[3.5,6])
        fig.subplots_adjust(left=0.17, bottom=0.1, hspace=0.15,
                            right=0.97, top=0.97)

    fig.clf()
    def line(x, slope, zpt):
        return slope*x + zpt
    nw, se = lightcurve.get_spock_data()

    outfilename = os.path.join(__THISDIR__, 'data/magpk_trise_tfall.dat')
    fout = open(outfilename,'w')
    print("# event band deltatpk   mpk  fnupk  trise  t2   t3", file=fout)
    for event in ['nw','se']:
        if event.lower()=='se':
            sn = se
            mjdpkobs = __MJDPKSE__
            mjdprepk0, mjdpostpk0 = __MJDPREPK0SE__, __MJDPOSTPK0SE__
            iax = 2
        else:
            sn = nw
            mjdpkobs = __MJDPKNW__
            mjdprepk0, mjdpostpk0 = __MJDPREPK0NW__, __MJDPOSTPK0NW__
            iax = 1
        if figsize=='tall':
            ax = pl.subplot(2,1,iax)
        else:
            ax = pl.subplot(1,2,iax)
        ax.invert_yaxis()

        # plot all measured magnitudes
        lightcurve.plot_lightcurve(src=event, aperture=np.inf,
                                   showtemplates=False, units='mag')

        # NOTE: the rest-frame time is always defined relative to the
        #  *observed* MJD of peak brightness, not the assumed mjdpk
        mjd = sn['MJD']
        trest = (mjd-mjdpkobs)/(1+__Z__)
        tprepk0 = (mjdprepk0-mjdpkobs)/(1+__Z__)
        tpostpk0 = (mjdpostpk0-mjdpkobs)/(1+__Z__)

        # fit a line to two well-sampled bands for each event,
        maginterpdict = {}
        for band in linfitbands:
            ib = np.where((sn['FILTER']==band.upper()) &
                          (trest>-3) & (trest<1))[0]
            if len(ib) == 0:
                continue
            m = sn['MAG'][ib]
            merr = sn['MAGERR'][ib]
            t = trest[ib]
            popt, pcov = scopt.curve_fit(line, t, m, sigma=merr, p0=[-0.5, 50])
            maginterpdict[band] = (popt, pcov)

        for band, color, marker, ls in zip(['f435w', 'f606w', 'f814w',
                                            'f105w', 'f125w', 'f140w', 'f160w'],
                                           ['c', 'b', 'darkgreen',
                                            'darkorange','r', 'm', 'darkred'],
                                           ['^', '>', 'v',
                                            's', 'd', 'h', 'o'],
                                           ['--',':','-',
                                            ':','--','-.','-']):
            ib = np.where((sn['FILTER']==band.upper()) &
                          (trest>-3) & (trest<1))[0]
            if len(ib) == 0:
                continue
            m = sn['MAG'][ib]
            merr = sn['MAGERR'][ib]
            t = trest[ib]
            magzpt = sn['ZP'][ib[0]]

            # plot rise-time linear fits
            if band not in linfitbands:
                continue
            tplotrise = np.array([-3, tpostpk0 - 0.5])
            riseslope, risezpt = maginterpdict[band][0]
            magrise = riseslope * tplotrise + risezpt
            ax.plot(tplotrise, magrise, marker=' ', ls=ls, color=color)

            # plot decline-time linear fits
            # We arbitrarily set the "0-flux" magnitude to 30
            # mag0 = riseslope * tprepk0 + risezpt  # alternative definition
            mag0 = 30
            for deltatpk in np.arange(tprepk0,tpostpk0,0.1):
                magpk = riseslope * deltatpk + risezpt
                declineslope = (mag0 - magpk) / (tpostpk0-deltatpk)
                declinezpt = magpk - declineslope * deltatpk
                m3 = magpk + 3
                m2 = magpk + 2
                t3 = min((tpostpk0 - deltatpk),
                         ((m3 - declinezpt) / declineslope) - deltatpk)
                t2 = min((tpostpk0 - deltatpk),
                         ((m2 - declinezpt) / declineslope) - deltatpk)
                trise = deltatpk - tprepk0
                fnu_uJy = 10**((23.9-magpk)/2.5) # flux density in microJanskys
                print("%s  %s  %5.2f  %5.2f  %7.4f  %5.3f  %5.3f  %5.3f" % (
                    event, band, deltatpk, magpk, fnu_uJy, trise, t2, t3),
                    file=fout)
                if ((deltatpk == 0) or
                    (np.abs(deltatpk - (tpostpk0 - 0.5)) <= 0.05) or
                    (np.abs(deltatpk - (tpostpk0 / 2.)) <= 0.05)):
                    tplotdecline = np.array([deltatpk, tpostpk0])
                    mplotdecline = declineslope * tplotdecline + declinezpt
                    ax.plot(tplotdecline, mplotdecline, marker=' ', ls=ls,
                             color=color)
                    if declinetimemetric=='t3':
                        ax.plot(t3+deltatpk, m3, marker='o', mfc='w',mec=color, ms=8,
                                mew=1.5)
                    elif declinetimemetric=='t2':
                        ax.plot(t2+deltatpk, m2, marker='o', mfc='w',mec=color, ms=8,
                                mew=1.5)
                    ax.plot(deltatpk, magpk, marker='d', mfc='w',mec=color,
                            ms=8, mew=1.5)

        if event=='nw':
            ax.set_xlim(-6.5, 6.5)
            ax.set_ylim(30.2, 24.5)
            ax.text(0.95,0.9, 'NW', ha='right', va='top',
                    transform=ax.transAxes, fontsize='large')
            ax.legend(loc='upper left', fontsize='small', frameon=False)
            ax.set_ylabel('Observed AB magnitude')
        else:
            ax.set_xlim(-3.9, 8.9)
            ax.set_ylim(30.2, 22.8)
            ax.text(0.95,0.9, 'SE', ha='right', va='top',
                    transform=ax.transAxes, fontsize='large')
            ax.set_xlabel('Rest-frame time (days)')
            ax.set_ylabel('Observed AB magnitude')
            ax.plot(0.05,0.8,marker='o', mfc='w',mec='darkred',
                    transform=ax.transAxes)
            ax.plot(0.05,0.9,marker='d', mfc='w',mec='darkred',
                    transform=ax.transAxes)
            ax.text(0.08, 0.88, '($t_{\\rm pk}$, $m_{\\rm pk}$)',
                    transform=ax.transAxes,
                    ha='left', va='center', color='darkred')
            ax.text(0.08, 0.78, r'($t_{\rm pk}+t_{\rm 2}$, $m_{\rm pk}+2$)',
                    transform=ax.transAxes,
                    ha='left', va='center', color='darkred')

    fout.close()
    return

def plot_ps1_fast_transients(
        ax1, ax2, plotrisetime=False, declinetimemetric='t2',
        datfile=os.path.join(__THISDIR__,'data/drout2014_table4.dat')):
    """ plot the fast transients from Drout et al. 2014"""
    indat = ascii.read(datfile, format='commented_header',
                       header_start=-1, data_start=0)
    trise = (indat['t_rise_max'] + indat['t_rise_min'])/2.
    err_trise = indat['t_rise_max'] - trise

    if declinetimemetric=='t3':
        # time to decline by 3 mags, assuming a linear decline in mag
        #  t_x = t_1/2 * ( X / 2.5log10(2) )
        tdecline = (3 / (2.5*np.log10(2.))) * indat['t_decline']
    elif declinetimemetric=='t2':
        # time to decline by 2 mags, assuming a linear decline in mag
        tdecline = (2 / (2.5*np.log10(2.))) * indat['t_decline']
    elif declinetimemetric=='t1/2':
        # time to decline to half of peak flux, measured by Drout et al
        tdecline = indat['t_decline']

    err_tdecline = indat['err_t_decline']
    M = indat['Mrest']
    fpk = 10**(-0.4*(M-25))

    for bandname, color in zip(['g','r'],
                               ['g', 'darkorange']):
        band = sncosmo.get_bandpass('sdss'+bandname)
        wave_eff = band.wave_eff
        iband = np.where(indat['Band']==bandname)[0]
        logL = np.log10(__cL0__ / wave_eff) + 40 - (0.4 * M[iband])
        err_logL = 0.4 * indat['err_Mrest'][iband]
        if plotrisetime:
            ax1.errorbar(trise[iband], logL, yerr=err_logL,
                         xerr=err_trise[iband], mec=color,
                         color=color, ls=' ', marker='o')
        ax2.errorbar(tdecline[iband], logL, yerr=err_logL,
                     xerr=err_tdecline[iband], mec='k',
                     color=color, ls=' ', marker='o', ms=8)


def plot_classical_novae(declinetimemetric='t2', marker='D',
                         downes2000=True, shafter2011=True,
                         kasliwal2011=True, czekala2013=True):
    """ Plot squares for the classical novae from the MW and M31,
    using data from Downes & Duerbeck 2000, Shafter 2011, Kasliwal 2011,
    and Czekala 2013

    :param declinetimemetric: t2, t3, or t1/2
    :return:
    """

    if downes2000:
        downesdatfile = os.path.join(__THISDIR__,"data/downes2000_table5.dat")
        indat = ascii.read(downesdatfile,
                           format='commented_header', header_start=-1,
                           data_start=0)
        t3 = indat['t3']
        MV = indat['MVmax']
        errMV = indat['e_MVmax']
        wave_eff = 5500
        logLV = np.log10(__cL0__ / wave_eff) + 40 - (0.4 * MV)
        logLVerr = 0.4*errMV
        if declinetimemetric=='t3':
            tdecline = t3
        elif declinetimemetric=='t2':
            tdecline = t3 * (2 / 3.)
        elif declinetimemetric=='t1/2':
            tdecline = t3 / (3 / (2.5*np.log10(2.)))
        else:
            raise exceptions.RuntimeError("decline time must be t2, t3, or t1/2")
        pl.plot(tdecline , logLV,  marker=marker, mec='k',
                color='darkcyan', ls=' ', ms=8, label='_Downes & Duerbeck 2000')

    if czekala2013:
        # 3 very bright Classical novae from Czekala et al 2013
        for label, MV, errMV, t2, errt2 in zip(
                ['_L91', '_M31N-2007-11d', '_SN 2010U'],
                [-10, -9.5, -10.2],
                [0.1, 0.1, 0.1],
                [6, 9.5, 3.5],
                [0.5, 0.5, 0.3]):
            wave_eff = 5500
            logLV = np.log10(__cL0__ / wave_eff) + 40 - (0.4 * MV)
            logLVerr = 0.4 * errMV
            if declinetimemetric=='t3':
                tdecline = t2  * (3. / 2.)
            elif declinetimemetric=='t2':
                tdecline = t2
            elif declinetimemetric=='t1/2':
                tdecline = t2 / (2 / (2.5*np.log10(2.)))
            else:
                raise exceptions.RuntimeError(
                    "decline time must be t2, t3, or t1/2")

            pl.errorbar(tdecline, logLV, # logLVerr, errt2,
                        marker=marker, mec='k',
                        color='darkorange', ls=' ', ms=8, label=label)

    if kasliwal2011:
        # Kasliwal+ 2011
        Mg = np.array([-7.5, -8.7, -10.7, -7.6, -8.5, -7.8, -6.8, -7.6, -8.5,
                       -7.8, -7., -7.5, -6.5, -7.7])
        t2 = np.array([24.6, 6., 8., 8., 14., 16.6, 16, 12., 16.2, 2., 8.6,
                       26.3, 12.3, 7.5])
        wave_eff = 4718.0
        logLg = np.log10(__cL0__ / wave_eff) + 40 - (0.4 * Mg)
        pl.errorbar(t2, logLg, marker=marker, mec='k',
                    color='darkgreen', ls=' ', ms=8, label='_Kasliwal+ 2011')

    if shafter2011:
        datfile = os.path.join(__THISDIR__,"data/shafter2011_table6.dat")
        indat = ascii.read(datfile,
                           format='commented_header', header_start=-1,
                           data_start=0)

        for band,bandname,plotcolor in zip(
                ['B','V','R','darkred'],
                ['bessellb','bessellv','bessellr','sdssr'],
                ['b','darkcyan','darkorange','r']):
            wave_eff = sncosmo.get_bandpass(bandname).wave_eff
            iband = np.where(indat['Filter'] == band)
            M = indat['Mmax'][iband]
            errM = indat['errMmax'][iband]
            t2 = indat['t2'][iband]
            errt2 = indat['errt2'][iband]
            logL = np.log10(__cL0__ / wave_eff) + 40 - (0.4 * M)
            logLerr = 0.4 * errM
            if declinetimemetric == 't3':
                tdecline = t2 * (3 / 2.)
            elif declinetimemetric == 't2':
                tdecline = t2
            elif declinetimemetric == 't1/2':
                tdecline = t2 / (2 / (2.5 * np.log10(2.)))
            else:
                raise exceptions.RuntimeError(
                    "decline time must be t2, t3, or t1/2")
            #pl.errorbar(tdecline, logL, yerr=logLerr, xerr=errt2,
            pl.plot(tdecline, logL,
                    marker=marker, mec='k', color=plotcolor,
                    ls=' ', ms=8, label='_Shafter+ 2011')

    return


def plot_recurrent_novae(timemetric='t2', luminositymetric='Lpk',
                         marker='d', color='k',
                         plotbands = 'gBVR', ax=None):
    """
    plot the peak luminosity vs decline time for recurrent novae,
    using data from Schaefer 2010
    and the fast-recurrence RN from Darnley+ 2014
    """
    MB = np.array(
        [-7.0, -6.7, -7.6, -7.3, -8.4, -8.8, -7.7, -10.8, -8.1, -9.0])
    MV = np.array(
        [-7.1, -6.6, -7.1, -7.4, -8.5, -8.4, -7.6, -10.6, -8.2, -8.9])
    AmpV = np.array(
        [9.1, 9.8, 7.7, 8.2, 10.1, 11.2, 7.3, 6.2, 9.2, 7.4])
    AmpB = np.array(
        [8.9, 10.2, 8.0, 8.2, 10.3, 12.0, 7.9, 6.9, 9.5, 7.4])

    id = ['T Pyx', 'IM Nor', 'CI Aql',
          'V2487 Oph', 'U Sco', 'V394 CrA', 'T CrB',
          'RS Oph', 'V745 Sco', 'V3890 Sgr']
    trec_yrs = np.array([19, 82, 24, 18, 10.3, 30, 80, 14.7, 21, 25])
    t3 = np.array([62, 80.0, 31.6, 8.4, 2.6, 5.2, 6.0, 14.0, 9.0, 14.4])
    t2 = np.array([32, 50.0, 25.4, 6.2, 1.2, 2.4, 4.0, 6.8, 6.2, 6.4])

    if ax is None:
        ax = pl.gca()
    if timemetric== 't2':
        t = t2
    elif timemetric== 't3':
        t = t3
    elif timemetric== 't3':
        t = t3
    elif timemetric == 'Prec' or timemetric == 'trec':
        t = np.log10(trec_yrs)
    else:
        raise exceptions.RuntimeError('decline time must be t2, t3, or t1/2')

    wave_effV = 5500
    logLV = np.log10(__cL0__ / wave_effV) + 40 - (0.4 * MV)
    wave_effB = 4385
    logLB = np.log10(__cL0__ / wave_effB) + 40 - (0.4 * MB)

    if luminositymetric=='Lpk':
        yV = logLV
        yB = logLB
    elif luminositymetric.startswith('Amp'):
        yV = AmpV
        yB = AmpB

    if 'V' in plotbands:
        ax.plot(t, yV, marker=marker, color='k', ls=' ', ms=8)
    if 'B' in plotbands:
        ax.plot(t, yB, marker=marker, color='blue', ls=' ', ms=8)

    # 2014 eruption of the fast-recurrence nova M31N-2008-12a
    # Darnley et al 2015
    if 'V' in plotbands:
        t3V = 3.8 # Darnley et al. 2015, Table 5
        # mV = 18.5 # +- 0.1 Hornoch et al 2014, Darnley et al 2015
        # convert to absolute mag using distance modulus to M31 of
        MV = -6.55 # Darnley et al. 2015, section 6.2
        wave_eff = 5550 # g band wavelength in Angstroms
        logLV = logLfromMV(MV)
        AmpV =  3.5
        AmpVerr = 1.0
        if timemetric== 't3':
            t = t3V
        elif timemetric== 't2':
            t = t3V * (2. / 3.)
        elif timemetric== 't1/2':
            t = t3V / (3 / (2.5*np.log10(2.)))
        elif timemetric=='Prec' or timemetric=='trec':
            t = np.log10(1.0) # recurrence time in years
        else:
            raise exceptions.RuntimeError("decline time must be t2, t3, or t1/2")
        if luminositymetric=='Lpk':
            ax.plot(t, logLV, marker='+', color=color,
                    ms=rcParams['lines.markersize']*2,
                    lw=rcParams['lines.linewidth']*1.3,
                    mew=rcParams['lines.linewidth']*2,
                    label='_M31N-2008-12a (V)')
        elif luminositymetric=='Amp':
            #ax.errorbar(t, AmpV, yerr=[[0],[AmpVerr]], lolims=True,
            #            marker=marker, color='darkcyan', ms=12,
            #            label='_M31N-2008-12a (V)')
            ax.errorbar(t, AmpV, marker='+', color=color,
                        ms=rcParams['lines.markersize']*2,
                        lw=rcParams['lines.linewidth']*1.3,
                        mew=rcParams['lines.linewidth']*2,
                        label='_M31N-2008-12a (V)')
            # ax.text(t*3, AmpV*0.5, 'M31N-2008-12a', color='darkcyan',
            #        fontsize='small', ha='left', va='top')

    if 'R' in plotbands:
        wave_eff = 6670 # R band wavelength in Angstroms
        t3R = 4.8 # Darnley et al. 2015, Table 5
        MR = -6.6 # Tang et al. 2014 (abstract)
        logLR = np.log10(__cL0__ / wave_eff) + 40 - (0.4 * MR)
        if timemetric== 't3':
            t = t3R
        elif timemetric== 't2':
            t = t3R * (2 / 3.)
        elif timemetric== 't1/2':
            t = t3R / (3 / (2.5*np.log10(2.)))
        elif timemetric=='Prec' or timemetric=='trec':
            t = np.log10(1.0) # recurrence time in years
        else:
            raise exceptions.RuntimeError("decline time must be t2, t3, or t1/2")
        ax.plot(t, logLR, marker=marker, color='darkorange', ms=12,
                label='_M31N-2008-12a (R)')
    return


def plot_yaron2005_models(
        timemetric='t2', luminositymetric='Amp', ax=None,
        datfile=os.path.join(__THISDIR__,'data/yaron2005_table3.dat'),
        markersizemetric='MWD', markercolormetric='logMdot',
        markersizelegend=True, markercolorlegend=True):
    """  plot the peak luminosity vs time (decline/recurrence) from
    Yaron et al 2005
    :param datfile: name of the data file with the Yaron+ 2005 data
    :return:
    """
    if ax is None:
        ax = pl.gca()
    indat = ascii.read(datfile, format='commented_header',
                       header_start=-1, data_start=0)
    # MWD  TWD logMdot vmax vavg L4max A t3bol tml Prec

    if timemetric== 't3':
        t = indat['t3bol']
    elif timemetric== 't2':
        t = indat['t3bol'] * 2. / 3.
    elif timemetric== 't1/2':
        t = indat['t3bol'] / (3 / (2.5*np.log10(2.)))
    elif timemetric in ['trec', 'Prec']:
        t = np.log10(indat['Prec'])
    else:
        raise exceptions.RuntimeError("decline time must be t2, t3, or t1/2")

    Amp = indat['A']
    L4max = indat['L4max'] * 0.4
    # the factor of 0.4 accounts for the correction from bolometric
    # luminosity to visual band luminosity (using BC=-0.1 mag from
    # Allen 1973 and Livio 1992)

    # convert bolometric luminosity from units of 10^4 Lsun to erg/s
    Lsun = 3.86e33 # ergs/s
    logLmax = np.log10(L4max * 1e4 * Lsun)
    if luminositymetric=='Amp':
        y = Amp
    else :
        y = logLmax

    if markersizemetric=='MWD':
        msmin = 0.5 * rcParams['lines.markersize']
        msmax = 2 * rcParams['lines.markersize']
        mwd = indat['MWD']
        ms = (msmax-msmin) * ((mwd-mwd.min())/(mwd.max()-mwd.min())) + msmin

        if markersizelegend:
            ax.plot(-1, -1, marker='o', ls=' ', ms=msmax, mfc='w', mec='k',
                    label='1.4')
            ax.plot(-1, -1, marker='o', ls=' ', ms=(msmax+msmin)/2.,
                    mfc='w', mec='k',  label='0.9')
            ax.plot(-1, -1, marker='o', ls=' ', ms=msmin, mfc='w', mec='k',
                    label='0.4')
    else:
        ms = [rcParams['ms'] for i in range(len(t))]

    if markercolormetric=='logMdot':
        logMdot = indat['logMdot']
        colorscale = (logMdot - logMdot.min()) / (logMdot.max()-logMdot.min())
        color = cm.Spectral(colorscale)
        if markercolorlegend:
            colormax = cm.Spectral(0.99)
            colormid = cm.Spectral(0.5)
            colormin = cm.Spectral(0.01)
            # import pdb; pdb.set_trace()
            ax.plot(-1, -1, marker='o', ls=' ',
                    ms=1.2*rcParams['lines.markersize'],
                    mfc=colormax, mec='k', alpha=1,
                    label='10$^{-6}$')
            ax.plot(-1, -1, marker='o', ls=' ',
                    ms=1.2*rcParams['lines.markersize'],
                    mfc=colormid, mec='k', alpha=1,
                    label='10$^{-9}$')
            ax.plot(-1, -1, marker='o', ls=' ',
                    ms=1.2*rcParams['lines.markersize'],
                    mfc=colormin, mec='k', alpha=1,
                    label='10$^{-12.3}$')
    else:
        color=['k' for i in range(len(t))]

    for i in range(len(t)):
        ax.plot(t[i], y[i], marker='o', ls=' ', ms=ms[i],
                color=color[i], mec='k', alpha=0.3, label='_')


def plot_mmrd(ax=None, declinetimemetric='t2', livio92=True,
              dellavalle95=True, tmax=18):
    """ plot the traditional peak luminosity vs decline time
    (i.e. the Max Mag vs Rate of Decline  (MMRD) relation)
    from Della Valle and Livio 1995 or Downes and Duerbeck 2000.
    i.e. the theoretical
    :return:
    """
    if ax is None:
        ax = pl.gca()

    # constant combining the speed of light with the standard
    # luminosity factor to convert from AB mags to Luminosity in erg/s
    wave_eff = 5488.9 # V band wavelength in Angstroms

    if livio92:
        # Livio 1992
        MB = np.arange(-10, -7.0, 0.01)
        t3_livio92 = 51.3 * 10**((MB+9.76)/10) * (
            10**(2*(MB+9.76)/30) - 10**(-2*(MB+9.76)/30))**1.5
        logL_livio92 = np.log10(__cL0__ / wave_eff) + 40 - (0.4 * MB)

        if declinetimemetric=='t3':
            tdecline = t3_livio92
        elif declinetimemetric=='t2':
            tdecline = t3_livio92 * 2. / 3.
        elif declinetimemetric=='t1/2':
            tdecline = t3_livio92 / (3 / (2.5*np.log10(2.)))
        else:
            raise exceptions.RuntimeError("decline time must be t2, t3, or t1/2")
        ax.plot(tdecline, logL_livio92, color='k', ls='--',
                label='Livio 1992')

    if dellavalle95:
        # Cappaccioli 1990 or Della Valle and Livio 1995:
        # Mv = -7.92 - 0.81 arctan( (1.32 - log10( t2 ))/0.23 )
        # uncertainty is ~ +-0.5
        t2 = np.arange(0.01,tmax,0.05)
        MV_max = 0.45 - 7.92 - 0.81 * np.arctan((1.32 - np.log10(t2))/0.23)
        MV_min = -0.65 - 7.92 - 0.81 * np.arctan((1.32 - np.log10(t2))/0.23)
        logLmax = np.log10(__cL0__ / wave_eff) + 40 - (0.4 * MV_min)
        logLmin = np.log10(__cL0__ / wave_eff) + 40 - (0.4 * MV_max)
        if declinetimemetric=='t3':
            tdecline = t2 * 3. / 2.
        elif declinetimemetric=='t2':
            tdecline = t2
        elif declinetimemetric=='t1/2':
            tdecline = t2 / (2 / (2.5*np.log10(2.)))
        else:
            raise exceptions.RuntimeError("decline time must be t2, t3, or t1/2")

        ax.fill_between(tdecline, logLmin, logLmax, color='k', alpha=0.5, zorder=0)
        ax.fill_between(tdecline, logLmin-0.5, logLmax+0.5, color='k', alpha=0.2, zorder=-5)

    if False:
        # Downes and Duerbeck:
        # Mv = -8.02 - 1.23 arctan( (1.32 - log10( t2 ))/0.23 )
        MV_dd = -8.02 - 1.23 * np.arctan((1.32 - np.log10(t2))/0.23)
        logL_dd = np.log10(__cL0__ / wave_eff) + 40 - (0.4 * MV_dd)

        ax.plot(tdecline, logL_dd, color='k', alpha=0.5)
    return

import sncosmo
def plot_typeIa_sne(ax=None, showIax=True):
    """ make a crude representation of the Phillips relation
    for normal Type Ia and Iax
    :param ax:
    :return:
    """
    if ax is None:
        ax = pl.gca()
    MVmax = -16.5
    MVmin = -19.65
    t2min = 25
    t2max = 50
    t2 = [t2min, t2max]
    MV = np.array([MVmax, MVmin])
    MVtop = MV - 0.3
    MVbottom = MV + 0.3
    ax.fill_between(t2, logLfromMV(MVtop), logLfromMV(MVbottom),
                    color='k', alpha=0.3)

    if showIax:
        MVmax = -14.2
        MVmin = -18.9
        t2min = 25
        t2max = 50
        t2 = [t2min, t2max]
        MV = np.array([MVmax, MVmin])
        MVtop = MV - 1.0
        MVbottom = MV + 1.0
        ax.fill_between(t2, logLfromMV(MVtop),
                        logLfromMV(MVbottom),
                        color='m', alpha=0.3)


    return

def plot_kn_candidates(ax=None):
    """ plot limits for two kilonova candidates
    :return:
    """
    if ax == None:
        ax = pl.gca()

    # perley+ 2009
    logLpk = 41.2
    t2min = 6
    ax.plot(t2min, logLpk, marker='|', color='darkgreen',
            ms=rcParams['lines.markersize']*1.5)
    arr = FancyArrowPatch( [t2min, logLpk], [t2min+2,logLpk],
                           arrowstyle='-|>', mutation_scale=25,
                           fc='darkgreen', ls='solid')
    ax.add_patch(arr)

    # Berger+2013, Tanvir+2013
    zkn = 0.356
    mABpk = 20.5
    MABpk = mABpk - cosmo.mu(zkn) - 2.5 * np.log10(1 + zkn)

    logLpkmin = logLfromMV(MABpk)
    t2 = 0.578
    ax.plot(t2, logLpkmin, marker='_', color='darkblue',
            ms=rcParams['lines.markersize']*1.5)

    arr2 = FancyArrowPatch([t2, logLpkmin], [t2, logLpkmin+0.7],
                           arrowstyle='-|>', mutation_scale=25,
                           fc='darkblue', ls='solid')
    ax.add_patch(arr2)




def plot_spock(ax1, ax2=None, mumin=10, mumax=100, declinetimemetric='t2',
               plotrisetime=False):

    if ax2 is None:
        ax2 = ax1

    magdatfile = os.path.join(__THISDIR__,'data/magpk_trise_tfall_kcor_ab.dat')
    indat = ascii.read(magdatfile, format='commented_header',
                        header_start=-1, data_start=0)

    distmod = cosmo.mu(__Z__, H0=__H0__, Om=__OM__, Ode=1-__OM__)

    MABmin = indat['mpk'] - distmod - indat['kcor'] + 2.5 * np.log10(mumax)
    MABmax = indat['mpk'] - distmod - indat['kcor'] + 2.5 * np.log10(mumin)

    obsbandlist = np.unique(indat['obsband'])
    for obsbandname, label, c, m in zip(['f435w', 'f814w', 'f125w', 'f160w'],
                                        ['u','g','r','i'],
                                        ['c', 'darkgreen','r','darkred'],
                                        ['o','o', 's','s']):
        restbandname = __ABRESTBANDNAME__[obsbandname.lower()]

        restband = sncosmo.get_bandpass(restbandname)
        wave_eff = restband.wave_eff
        iplot = np.where((indat['obsband'] == obsbandname) &
                         (indat['deltatpk'] >= 0))[0]

        # computing log10(L) with luminosity in erg / s :
        logLmin = np.log10(__cL0__ / wave_eff) + 40 - (0.4 * MABmin[iplot])
        logLmax = np.log10(__cL0__ / wave_eff) + 40 - (0.4 * MABmax[iplot])

        if plotrisetime:
            ax1.fill_between(indat['trise'][iplot], logLmin, logLmax,
                             color=c, alpha=0.3, label=label)

        if declinetimemetric=='t2':
            tdecline = indat['t2'][iplot]
        elif declinetimemetric=='t3':
            tdecline = indat['t3'][iplot]
        elif declinetimemetric=='t1/2':
            tdecline = indat['t2'][iplot] / (2 / (2.5*np.log10(2.)))
        ax2.fill_between(tdecline, logLmin, logLmax,
                         color=c, alpha=0.3, label=label)



def mk_nova_comparison_figure(mumin=10, mumax=100, declinetimemetric='t2',
                              plotrisetime=False):
    """ Read in the data file giving apparent magnitude vs time inferred from
    the linear fits to four observed bands.  Read in the data file giving the
    K correction as a function of time for converting each observed HST band
    into the nearest rest-frame filter.  For each assumed time of peak,
    convert the extrapolated apparent mag at peak to the peak absolute
    magnitude.  Then convert this absolute magnitude to Luminosity in erg/s

    :return:
    """
    if plotrisetime:
        fig = plotsetup.fullpaperfig(figsize=[8, 3.5])
        fig.clf()
        ax1 = fig.add_subplot(1,2,1)
        ax2 = fig.add_subplot(1,2,2, sharey=ax1)
    else:
        fig = plotsetup.halfpaperfig(figsize=[4, 4])
        fig.clf()
        ax2 = fig.add_subplot(1,1,1)
        ax1 = ax2


    if plotrisetime:
        ax1.set_xlabel('t$_{\\rm rise}$: max rise time from 0 flux (days)')
        pl.setp(ax2.get_yticklabels(), visible=False)
        pl.setp(ax1.get_xticklabels()[-1], visible=False)
        fig.subplots_adjust(left=0.08, right=0.97, bottom=0.16, top=0.97,
                            wspace=0)
        ax1.set_xlim(0, 15)
    else:
        fig.subplots_adjust(left=0.16, right=0.97, bottom=0.16, top=0.97,
                            wspace=0)

    plot_spock(ax1, ax2, mumin=mumin, mumax=mumax,
               declinetimemetric=declinetimemetric, plotrisetime=plotrisetime)

    plot_mmrd(declinetimemetric=declinetimemetric,
              livio92=False, dellavalle95=True)
    # plot_yaron2005_models(declinetimemetric=declinetimemetric)
    plot_classical_novae(declinetimemetric=declinetimemetric, marker='D')
    plot_recurrent_novae(timemetric=declinetimemetric, marker='o')
    plot_kn_candidates(ax1)

    mmrd = ax2.plot(-1, -1, ls='-', marker=' ', lw=5, alpha=0.5,
                    color='k', label='Max. Mag - Rate of Decline')
    RNe = ax2.plot(-1, -1, ls=' ', marker='o', mfc='w', mec='k', ms=10,
                   label='Recurrent Novae')
    CNe = ax2.plot(-1, -1, ls=' ', marker='D', mfc='w', mec='k', ms=10,
                   label='Classical Novae')
    KNe = ax2.plot(-1, -1, ls='-', marker='>', mfc='w',
                   color='k', mec='k', ms=10,
                   label='Kilonova Candidates')

    #plot_ps1_fast_transients(ax1, ax2, plotrisetime=plotrisetime,
    #                         declinetimemetric=declinetimemetric)

    #ax1.text(8, 43, 'PS1 Fast Transients', ha='left', va='top',
    #         fontsize='small')
    #ax2.text(13.5, 41.8, 'Fast Transients', ha='right', va='top',
    #         fontsize='medium')

    ax2.text(5, 42.0, 'HFF14Spo', ha='right', va='top',
             fontsize='medium', color='k', rotation=-32)

    #ax2.text(13, 38.2, 'Novae', ha='right', va='top',
    #         fontsize='medium', color='k')
    ax2.legend(loc='upper right', fontsize='small', handlelength=1.5)

    ax2.set_xlim(0.01, 15.2)
    ax2.set_ylim(37.5, 44.5)

    # make the M_V axis on the right side
    MV = lambda logL: -2.5*(logL - np.log10(__cL0__ / 5500) - 40)
    logLlim = ax2.get_ylim()
    ax2right = ax2.twinx()
    ax2right.set_ylim(MV(logLlim[0]), MV(logLlim[1]))

    pl.setp(ax2.get_xticklabels()[0], visible=False)

    ax1.set_ylabel('log(L$_{\\rm pk}$ [erg/s])', labelpad=0)
    if declinetimemetric=='t3':
        ax2.set_xlabel('t$_{3}$: time to decline by 3 mag (days)')
    elif declinetimemetric=='t2':
        ax2.set_xlabel('t$_{2}$: time to decline by 2 mag (days)')
    ax2right.set_ylabel('$M_V$ at peak', labelpad=15, rotation=-90)

    ax2right.yaxis.set_major_locator(ticker.MultipleLocator(4))
    ax2right.yaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax2right.tick_params(pad=4)
    ax2.tick_params(pad=3)

    fig.subplots_adjust(left=0.14, bottom=0.15,
                        right=0.85, top=0.97)
    pl.draw()


def mk_sn_comparison_figure(showspock=True, mumin=10, mumax=100,
                            declinetimemetric='t2'):
    """ Plot a wide range of peak luminosities vs decline times
    :return:
    """
    fig = plotsetup.halfpaperfig(figsize=[4, 4])
    fig.clf()
    ax = fig.add_subplot(1,1,1)
    ax2 = ax.twinx()

    if showspock:
        plot_spock(ax, ax2=None, mumin=mumin, mumax=mumax,
                   declinetimemetric=declinetimemetric, plotrisetime=False)
        ax.text(0.6, 42.3, 'HFF14Spo', ha='left', va='top',
                fontsize='small', color='k', rotation=-20)

    plot_mmrd(ax, declinetimemetric=declinetimemetric, livio92=False,
              dellavalle95=True, tmax=120)
    plot_typeIa_sne(ax)
    ax2.text(87, -18.5, 'Type Ia SNe', rotation=53,
             ha='center', va='center',
             fontsize='small', color='k', zorder=1000)
    ax2.text(90, -16.5, 'Type Iax', rotation=62,
             ha='center', va='center',
             fontsize='small', color='darkmagenta', zorder=1000)

    ax.text(9, 38.7, 'Novae', ha='right', va='top',
             fontsize='small', color='k')

    ax.set_xlim(0.5,115)
    ax.set_ylim(logLfromMV(-5.2), logLfromMV(-24.5))
    ax.set_xscale('log')


    # make the M_V axis on the right side
    logLlim = ax.get_ylim()
    ax2.set_xlim(0.5,115)
    ax2.set_ylim(MVfromLogL(logLlim[0]), MVfromLogL(logLlim[1]))

    pl.setp(ax.get_xticklabels()[0], visible=False)

    ax.set_ylabel('log(L$_{\\rm pk}$ [erg/s])', labelpad=0)
    ax.set_xlabel('t$_2$ : time to decline by 2 mag (days)')
    ax2.set_ylabel('$M_V$ at peak', labelpad=15, rotation=-90)

    ax.yaxis.tick_left()

    ps1ft = Ellipse(xy=(67,-18.5), width=25, height=3, angle=0,
                    facecolor='b', alpha=0.3)
    #ax2.text(67, -18.5, 'Fast\nOptical\nTransients',
    ax2.text(54, -19, 'Fast Optical\nTransients',
             ha='right', va='bottom',
             fontsize='small', color='darkblue')

    crt = Ellipse(xy=(78,-14), width=25, height=5, angle=0,
                  facecolor='darkorange', alpha=0.5, zorder=30)
    ax2.text(77, -13, 'Ca-Rich\nSNe',
             ha='center', va='center',
             fontsize='small', color='k')

    lrn = Ellipse(xy=(97,-10.3), width=30, height=7, angle=0,
                  facecolor='darkred', alpha=0.3, zorder=40)
    ax2.text(98, -10.3, 'Luminous\nRed\nNovae', ha='center', va='center',
             fontsize='small', color='darkred')

    ccsn = Rectangle(xy=(75,-20), width=45, height=6, angle=0,
                     facecolor='darkcyan', alpha=0.3, zorder=20)
    ax2.text(101, -15.5, 'Core\nCollapse\nSNe', ha='center', va='center',
             fontsize='small', color='darkcyan')

    slsn = Rectangle(xy=(90,-23), width=30, height=2, angle=0,
                     facecolor='darkorchid', alpha=0.3)
    ax2.text(88, -22, 'Superluminous\nSupernovae',
             ha='right', va='center',
             fontsize='small', color='darkorchid')

    ptIa = Ellipse(xy=(58,-16.5), width=30, height=3, angle=0,
                   facecolor='w', ls='dashed', edgecolor='k',
                   zorder=40, alpha=0.3)
    ax2.text(58, -16.5, '.Ia', ha='center', va='center',
             fontsize='small', color='k', zorder=1000)

    #kNe = Ellipse(xy=(38,-14.), width=40, height=3, angle=0,
    #               facecolor='w', ls='dashed', edgecolor='k',
    #               zorder=40, alpha=0.3)
    kNe = Rectangle(xy=(9,-14), width=50, height=3, angle=0,
                    facecolor='w', ls='dashed', edgecolor='k',
                    alpha=0.3)
    ax2.text(35, -12, 'Kilonovae', ha='center', va='center',
             fontsize='small', color='k', zorder=1000)
    ax2.tick_params(pad=4)
    ax.tick_params(pad=3)

    for shape in [ps1ft, crt, lrn, ccsn, slsn, ptIa, kNe]:
        ax2.add_artist(shape)


    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    ax2.yaxis.set_major_locator(ticker.MultipleLocator(4))
    ax2.yaxis.set_minor_locator(ticker.MultipleLocator(1))

    fig.subplots_adjust(left=0.13, bottom=0.15,
                        right=0.83, top=0.97)
    pl.draw()


def mk_amplitude_vs_prec_fig():
    fig = plotsetup.fullpaperfig()
    # derive a lower limit on the spock outburst amplitudes by comparing the
    #  peak mag against the arbitrary baseline magnitude (AB mag = 30)
    AmpSpock = 5
    AmpSpockErr = 2
    trec_yrs_spock = np.log10(150 / 365.)

    ax1 = fig.add_subplot(1,2,1)
    ax2 = fig.add_subplot(1,2,2, sharex=ax1)
    for lolims, marker, zorder in zip([2, 0],['D',' '],[2000,-100]):
        #ax1.errorbar([trec_yrs_spock1,], [AmpSpock1mid,],
        #             [[AmpSpock1err],[AmpSpock1err*lolims]],
        #             lolims=lolims, marker=marker, color='darkgreen',
        #             alpha=0.5, ls=' ')
        ax1.errorbar(trec_yrs_spock, AmpSpock,
                     [[AmpSpockErr],[AmpSpockErr*lolims]],
                     lolims=lolims, marker=marker,
                     alpha=1, ls=' ', color='darkred',
                     lw=rcParams['lines.linewidth']*1.3,
                     ms=rcParams['lines.markersize']*1.5,
                     mec='darkred', mfc='w', zorder=zorder)

    # plot RN from MW and M31:
    plot_yaron2005_models(timemetric='trec', luminositymetric='Amp', ax=ax1,
                          markersizelegend=True, markercolorlegend=False)
    plot_recurrent_novae('trec', 'Amp', marker='x', plotbands=['V'], ax=ax1)
    ax1.set_xlim(-2.5, 9)
    ax1.set_ylim(0.1, 25.0)
    # ax1.set_xscale('log')

    pl.setp(ax1.get_xticklabels()[0], visible=False)

    ax1.set_xlabel('log$_{10}$( Recurrence Period [years])', labelpad=0)
    ax1.set_ylabel('Outburst Amplitude (mag)', labelpad=0)
    ax1.tick_params(pad=3)

    # ---------------------------------------------------
    # Spock:
    # Precurrence = 3 - 5 months
    logtrecuryr = np.log10(4 / 12.)
    # Peak Luminosity = 10^41 erg / s
    # err_trec = np.
    logLpk = 41
    errlogLpk = 1
    ax2.errorbar(logtrecuryr, logLpk,
                 yerr=errlogLpk, marker='D', color='darkred',
                 ms=rcParams['lines.markersize']*1.5,
                 mec='darkred', mfc='w')

    plot_yaron2005_models(timemetric='trec', luminositymetric='Lpk',
                          ax=ax2, markersizelegend=False,
                          markercolorlegend=True)
    plot_recurrent_novae(ax=ax2, timemetric='trec', marker='x',
                         plotbands=['V'])

    ax1.legend(loc='lower right', markerscale=1,
               title='M$_{\\rm WD}$ (M$_{\\odot}$)',
               borderaxespad=0.8, fontsize='small',
               handletextpad=0.8, frameon=False)

    ax2.legend(loc='upper right', markerscale=1,
               title='$\dot{\\rm M}$ (M$_{\\odot}$ yr$^{-1}$)',
               borderaxespad=0.8, fontsize='small',
               handletextpad=0.8, frameon=False)

    ax2.set_ylim(36.8, 42.4)
    # ax2.set_xscale('log')
    axright = ax2.twinx()
    logLlim = ax2.get_ylim()
    axright.set_ylim(MVfromLogL(logLlim[0]), MVfromLogL(logLlim[1]))
    pl.setp(ax2.get_xticklabels()[0], visible=False)
    ax2.set_ylabel('log$_{10}$($\\nu {\\rm L}_{\\nu,{\\rm pk}}$ [erg/s])', labelpad=0)
    ax2.set_xlabel('log$_{10}$( Recurrence Period [years])', labelpad=0)
    axright.set_ylabel('$M_V$ at peak', labelpad=15, rotation=-90)
    axright.yaxis.set_major_locator(ticker.MultipleLocator(4))
    axright.yaxis.set_minor_locator(ticker.MultipleLocator(1))
    axright.tick_params(pad=4)
    ax2.tick_params(pad=3)

    fig = pl.gcf()
    fig.subplots_adjust(left=0.07, bottom=0.15, wspace=0.25,
                        right=0.92, top=0.94)
    pl.draw()

