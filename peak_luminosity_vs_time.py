from __future__ import print_function
from constants import __MJDPKNW__, __MJDPKSE__, __Z__
from constants import __MJDPREPK0NW__, __MJDPOSTPK0NW__
from constants import __MJDPREPK0SE__, __MJDPOSTPK0SE__
from constants import __RESTBANDNAME__, __H0__ , __OM__

from . import lightcurve
from scipy import interpolate as scint
from scipy import optimize as scopt
import numpy as np
from matplotlib import pyplot as pl
from pytools import plotsetup, cosmo
from astropy.io import ascii
import sncosmo

import sys
import os

__THISFILE__ = sys.argv[0]
if 'ipython' in __THISFILE__:
    __THISFILE__ = __file__
__THISDIR__ = os.path.abspath(os.path.dirname(__THISFILE__))

def linear_fit_light_curves(linfitbands=['f435w', 'f814w', 'f125w', 'f160w'],
                            figsize='tall'):
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
                    #ax.plot(t3+deltatpk, m3, marker='o', mfc='w',mec=color, ms=8,
                    #        mew=1.5)
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
        ax1, ax2, plotrisetime=False,
        datfile=os.path.join(__THISDIR__,'data/drout2014_table4.dat')):
    """ plot the fast transients from Drout et al. 2014"""
    indat = ascii.read(datfile, format='commented_header',
                       header_start=-1, data_start=0)
    trise = (indat['t_rise_max'] + indat['t_rise_min'])/2.
    err_trise = indat['t_rise_max'] - trise

    t3 = 1.874 * indat['t_decline']
    err_t3 = indat['err_t_decline']
    M = indat['Mrest']
    fpk = 10**(-0.4*(M-25))

    # f_at_t3 = fpk * 0.0631  # Flux after it has declined by 3 mags

    # MWD  TWD logMdot vmax vavg L4max A t3bol tml Prec

    # constant combining the speed of light with the standard
    # luminosity factor to convert from AB mags to Luminosity in erg/s
    cL0 = 0.13027455 # x10^40 erg s-1 Angstrom   (the 10^40 is handled below)

    for bandname, color in zip(['g','r'],
                               ['g', 'darkorange']):
        band = sncosmo.get_bandpass('sdss'+bandname)
        wave_eff = band.wave_eff
        iband = np.where(indat['Band']==bandname)[0]
        logL = np.log10(cL0 / wave_eff) + 40 - (0.4 * M[iband])
        err_logL = 0.4 * indat['err_Mrest'][iband]
        if plotrisetime:
            ax1.errorbar(trise[iband], logL, yerr=err_logL,
                         xerr=err_trise[iband], mec=color,
                         color=color, ls=' ', marker='o')
        ax2.errorbar(t3[iband], logL, yerr=err_logL,
                     xerr=err_t3[iband], mec='k',
                     color=color, ls=' ', marker='o', ms=8)


def plot_known_novae():

    # M31N200812a
    distmodM31 = 24.4  # (Vilardelli et al. 2010)
    EBVmin = 0.1
    EBVmax = 0.26

    # galactic extinction
    AR = 0.134
    Ag = 0.205

    # total LOS extinction
    AR = 0.45 # +-0.23
    Ag = 0.7 # +- 0.35

    # 2011 outburst
    g = 18.51
    R = 18.18
    MR = -6.5

    # 2013 outburst
    mR = 18.34
    MR = -6.7

    # 2014 eruption
    # Darnley et al 2015
    t3V = 3.8 # Darnley et al. 2015, Table 5
    # mV = 18.5 # +- 0.1 Hornoch et al 2014, Darnley et al 2015
    MV = 18.5 - 0.7 - 24.4

    t3R = 4.8 # Darnley et al. 2015, Table 5
    MR = -6.6 # Tang et al. 2014 (abstract)

    cL0 = 0.13027455 # x10^40 erg s-1 Angstrom   (the 10^40 is handled below)

    wave_eff = 5550 # g band wavelength in Angstroms
    logLV = np.log10(cL0 / wave_eff) + 40 - (0.4 * MV)
    pl.plot(t3V, logLV, marker='D', color='darkcyan', ms=12)

    wave_eff = 6670 # R band wavelength in Angstroms
    logLR = np.log10(cL0 / wave_eff) + 40 - (0.4 * MR)
    pl.plot(t3R, logLR, marker='D', color='darkorange', ms=12)

    downesdatfile = os.path.join(__THISDIR__,"data/downes2000_table5.dat")
    indat = ascii.read(downesdatfile,
                       format='commented_header', header_start=-1,
                       data_start=0)
    t3 = indat['t3']
    MV = indat['MVmax']
    errMV = indat['e_MVmax']
    wave_eff = 5500
    logLV = np.log10(cL0 / wave_eff) + 40 - (0.4 * MV)
    logLVerr = 0.4*errMV
    pl.errorbar( t3, logLV, logLVerr, marker='s', mec='k',
                 color='darkcyan', ls=' ', ms=8)

def plot_RNe():
    """
    plot the peak luminosity vs decline time for recurrent novae,
    using data from Schaefer 2010
    """
    MB = np.array(
        [-7.0, -6.7, -7.6, -7.3, -8.4, -8.8, -7.7, -10.8, -8.1, -9.0])
    MV = np.array(
        [-7.1, -6.6, -7.1, -7.4, -8.5, -8.4, -7.6, -10.6, -8.2, -8.9])
    id = ['T Pyx', 'IM Nor', 'CI Aql',
          'V2487 Oph', 'U Sco', 'V394 CrA', 'T CrB',
          'RS Oph', 'V745 Sco', 'V3890 Sgr']
    trec = np.array([19, 82, 24, 18, 10.3, 30, 80, 14.7, 21, 25])
    t3 = np.array([62, 80.0, 31.6, 8.4, 2.6, 5.2, 6.0, 14.0, 9.0, 14.4])
    t2 = np.array([32, 50.0, 25.4, 6.2, 1.2, 2.4, 4.0, 6.8, 6.2, 6.4])

    cL0 = 0.13027455 # x10^40 erg s-1 Angstrom   (the 10^40 is handled below)
    wave_effV = 5500
    logLV = np.log10(cL0 / wave_effV) + 40 - (0.4 * MV)

    wave_effB = 4385
    logLB = np.log10(cL0 / wave_effB) + 40 - (0.4 * MB)

    ax = pl.gca()
    ax.plot(t3, logLV, marker='d', color='darkcyan', ls=' ', ms=8)
    ax.plot(t3, logLB, marker='d', color='blue', ls=' ', ms=8)


def plot_yaron2005_models(
        datfile=os.path.join(__THISDIR__,'data/yaron2005_table3.dat')):
    """  plot the peak luminosity vs decline time from Yaron et al 2005
    :param datfile:
    :return:
    """
    indat = ascii.read(datfile, format='commented_header',
                       header_start=-1, data_start=0)
    # MWD  TWD logMdot vmax vavg L4max A t3bol tml Prec

    t3 = indat['t3bol']
    MV = indat['A']
    L4max = indat['L4max'] * 0.4
    # the factor of 0.4 accounts for the correction from bolometric
    # luminosity to visual band luminosity (using BC=-0.1 mag from
    # Allen 1973 and Livio 1992)

    # convert bolometric luminosity from units of 10^4 Lsun to erg/s
    Lsun = 3.86e33 # ergs/s
    logLmax = np.log10(L4max * 1e4 * Lsun)

    # cL0 = 0.13027455 # x10^40 erg s-1 Angstrom   (the 10^40 is handled below)
    # wave_eff = 5488.9 # V band wavelength in Angstroms
    # logLmaxFactor = np.log10(cL0 / wave_eff) + 40 - (0.4 * MV)
    pl.plot(t3, logLmax, marker='o', ls=' ', ms=8, color='k', mfc='w')


def plot_mmrd():
    """ plot the traditional peak luminosity vs decline time
    (i.e. the Max Mag vs Rate of Decline  (MMRD) relation)
    from Della Valle and Livio 1995 or Downes and Duerbeck 2000.
    i.e. the theoretical
    :return:
    """
    # constant combining the speed of light with the standard
    # luminosity factor to convert from AB mags to Luminosity in erg/s
    cL0 = 0.13027455 # x10^40 erg s-1 Angstrom   (the 10^40 is handled below)
    wave_eff = 5488.9 # V band wavelength in Angstroms


    # Livio 1992
    MB = np.arange(-10, -8.5, 0.01)
    t3_livio92 = 51.3 * 10**((MB+9.76)/10) * (
        10**(2*(MB+9.76)/30) - 10**(-2*(MB+9.76)/30))**1.5
    logL_livio92 = np.log10(cL0 / wave_eff) + 40 - (0.4 * MB)
    pl.plot(t3_livio92, logL_livio92, color='k', ls='--')

    # Cappaccioli 1990 or Della Valle and Livio 1995:
    # Mv = -7.92 - 0.81 arctan( (1.32 - log10( t2 ))/0.23 )
    # uncertainty is ~ +-0.5
    t2 = np.arange(0.01,12,0.05)
    MV_max = 0.45 - 7.92 - 0.81 * np.arctan((1.32 - np.log10(t2))/0.23)
    MV_min = -0.65 - 7.92 - 0.81 * np.arctan((1.32 - np.log10(t2))/0.23)

    # Downes and Duerbeck:
    # Mv = -8.02 - 1.23 arctan( (1.32 - log10( t2 ))/0.23 )
    # MV_dd = -8.02 - 1.23 * np.arctan((1.32 - np.log10(t2))/0.23)

    logLmax = np.log10(cL0 / wave_eff) + 40 - (0.4 * MV_min)
    logLmin = np.log10(cL0 / wave_eff) + 40 - (0.4 * MV_max)

    t3 = 1.3*t2
    # pl.fill_between( t2, MV_min, MV_max, color='k', alpha=0.5)
    pl.fill_between( t3, logLmax, logLmin, color='k', alpha=0.3)


def mk_figure(mumin=10,mumax=100, plotrisetime=False):
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

    magdatfile = os.path.join(__THISDIR__,'data/magpk_trise_tfall_kcor.dat')
    indat = ascii.read(magdatfile, format='commented_header',
                        header_start=-1, data_start=0)

    # constant combining the speed of light with the standard
    # luminosity factor to convert from AB mags to Luminosity in erg/s
    cL0 = 0.13027455 # x10^40 erg s-1 Angstrom   (the 10^40 is handled below)


    distmod = cosmo.mu(__Z__, H0=__H0__, Om=__OM__, Ode=1-__OM__)

    MABmin = indat['mpk'] - distmod - indat['kcor'] + 2.5 * np.log10(mumax)
    MABmax = indat['mpk'] - distmod - indat['kcor'] + 2.5 * np.log10(mumin)

    obsbandlist = np.unique(indat['obsband'])
    for obsbandname, label, c, m in zip(['f435w', 'f814w', 'f125w', 'f160w'],
                                        ['UV','B','r','i'],
                                        ['c', 'darkgreen','r','darkred'],
                                        ['o','o', 's','s']):
        restbandname = __RESTBANDNAME__[obsbandname.lower()]
        restband = sncosmo.get_bandpass(restbandname)
        wave_eff = restband.wave_eff
        iband = np.where(indat['obsband']==obsbandname)[0]

        # computing log10(L) with luminosity in erg / s :
        logLmin = np.log10(cL0 / wave_eff) + 40 - (0.4 * MABmin[iband])
        logLmax = np.log10(cL0 / wave_eff) + 40 - (0.4 * MABmax[iband])

        if plotrisetime:
            ax1.fill_between(indat['trise'][iband], logLmin, logLmax,
                             color=c, alpha=0.3, label=label)
        ax2.fill_between(indat['t3'][iband], logLmin, logLmax,
                         color=c, alpha=0.3, label=label)

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

    ax1.set_ylabel('log\,(\,L$_{\\rm pk}$~ [erg/s]\,)')
    ax2.set_xlabel('t$_{3}$: time to decline by 3 mag (days)')

    plot_mmrd()
    # plot_yaron2005_models()
    plot_known_novae()
    plot_RNe()
    plot_ps1_fast_transients(ax1, ax2, plotrisetime=plotrisetime)

    #ax1.text(8, 43, 'PS1 Fast Transients', ha='left', va='top',
    #         fontsize='small')
    ax2.text(8, 43.2, 'Fast Transients', ha='right', va='top',
             fontsize='medium')

    ax2.text(6, 41.8, 'HFF14Spo', ha='right', va='top',
             fontsize='medium', color='k', rotation=-20)

    ax2.text(13, 38.2, 'Novae', ha='right', va='top',
             fontsize='medium', color='k')

    ax2.set_xlim(-0.2,15.2)
    ax2.set_ylim(37.5, 43.5)
    # ax2.semilogx()
    pl.setp(ax2.get_xticklabels()[0], visible=False)
    pl.draw()

