from constants import __MJDPKNW__, __MJDPKSE__, __Z__
from constants import __MJDPREPK0NW__, __MJDPOSTPK0NW__
from constants import __MJDPREPK0SE__, __MJDPOSTPK0SE__
from . import lightcurve
from scipy import interpolate as scint
from scipy import optimize as scopt
import numpy as np
from matplotlib import pyplot as pl
from pytools import plotsetup
from astropy.io import ascii
import sncosmo

def linear_fit_light_curves(linfitbands=['f435w', 'f814w', 'f125w', 'f160w'],
                            figsize='tall'):
    """ Fit the rise and decline of the light curve with
     a straight line in mag vs time.
    """
    assert figsize in ['wide','tall']
    if figsize=='wide':
        fig = plotsetup.fullpaperfig(figsize=[8,3])
    else:
        fig = plotsetup.halfpaperfig()
    fig.clf()
    def line(x, slope, zpt):
        return slope*x + zpt
    nw, se = lightcurve.get_spock_data()

    fout = open('data/magpk_trise_tfall.dat','w')
    print >> fout, "# event band deltatpk   mpk  fnupk  trise  t3"
    for event in ['se','nw']:
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
            mag0 = riseslope * tprepk0 + risezpt # "0-flux" magnitude
            mag0 = 30
            for deltatpk in np.arange(0,tpostpk0,0.1):
                magpk = riseslope * deltatpk + risezpt
                declineslope = (mag0 - magpk) / (tpostpk0-deltatpk)
                declinezpt = magpk - declineslope * deltatpk
                tplotdecline = np.array([deltatpk, tpostpk0])
                magdecline = declineslope * tplotdecline + declinezpt
                m3 = magpk + 3
                t3 = min(tpostpk0,(magpk + 3 - declinezpt)/declineslope)
                trise = deltatpk - tprepk0
                fnu_uJy = 10**((23.9-magpk)/2.5) # flux density in microJanskys
                print >> fout,  "%s  %s  %5.2f  %5.2f  %7.4f  %5.3f  %5.3f" % (
                    event, band, deltatpk, magpk, fnu_uJy, trise, t3)
                if (deltatpk ==0 or np.abs(deltatpk-(tpostpk0-0.5))<=0.05 or
                    np.abs(deltatpk-(tpostpk0/2.))<=0.05):
                    ax.plot(tplotdecline, magdecline, marker=' ', ls=ls,
                             color=color)
                    ax.plot(t3, m3, marker='o', mfc='w',mec=color, ms=8,
                            mew=1.5)
                    ax.plot(deltatpk, magpk, marker='d', mfc='w',mec=color,
                            ms=8, mew=1.5)

        if event=='nw':
            ax.set_xlim(-6.5, 6.5)
            ax.set_ylim(30.2, 24.5)
            ax.text(0.95,0.9, 'NW', ha='right', va='top',
                    transform=ax.transAxes, fontsize='large')
            ax.legend(loc='upper left', fontsize='small')
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
            ax.text(0.08, 0.78, r'($t_{\rm pk}+t_{\rm 3}$, $m_{\rm pk}+3$)',
                    transform=ax.transAxes,
                    ha='left', va='center', color='darkred')

    fig.subplots_adjust(left=0.1, bottom=0.1, hspace=0.15,
                        right=0.95, top=0.95)
    fout.close()
    return


