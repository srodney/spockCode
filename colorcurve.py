from constants import __MJDPKNW__, __MJDPKSE__, __Z__
from . import lightcurve, kcorrections
# from scipy import interpolate as scint
from scipy import optimize as scopt
import numpy as np
from matplotlib import pyplot as pl
from pytools import plotsetup
from matplotlib import rcParams


def mk_color_curves_figure():
    fig = plotsetup.halfpaperfig()
    rcParams['text.usetex'] = False
    fig.clf()
    def line(x, slope, zpt):
        return slope*x + zpt
    nw, se = lightcurve.get_spock_data()
    for event in ['nw','se']:
        if event.lower()=='se':
            sn = se
            mjdpk = __MJDPKSE__
            ax1 = pl.subplot(2,2,2)
            ax2 = pl.subplot(2,2,4, sharex=ax1)
        else:
            sn = nw
            mjdpk = __MJDPKNW__
            ax1 = pl.subplot(2,2,1)
            ax2 = pl.subplot(2,2,3, sharex=ax1)

        mjd = sn['MJD']
        trest = (mjd-mjdpk)/(1+__Z__)
        tplot = np.arange(-3, +1, 0.1)
        colorset = []
        colorerrset = []

        # fit a line to one well-sampled band for each event,
        # between -3 and +1 rest-frame days from peak
        maginterpdict = {}
        for band in ['f814w', 'f125w', 'f160w']:
            ib = np.where((sn['FILTER']==band.upper()) &
                          (trest>-3) & (trest<1))[0]
            if len(ib) == 0:
                continue
            # m = np.append(np.array([31]), sn['MAG'][ib])
            # merr = np.append(np.array([0.5]), sn['MAGERR'][ib])
            m = sn['MAG'][ib]
            merr = sn['MAGERR'][ib]
            t = trest[ib]
            popt, pcov = scopt.curve_fit(line, t, m, sigma=merr, p0=[-0.5, 50])
            maginterpdict[band] = (popt, pcov)

        # plot all measured mags in the top panel
        for band, color, marker in zip(['f435w', 'f606w', 'f814w',
                               'f105w', 'f125w', 'f140w', 'f160w'],
                              ['c', 'b', 'darkgreen',
                                'darkorange','r', 'm', 'darkred'],
                              ['^', '>', 'v',
                               's', 'd', 'h', 'o']):
            ib = np.where((sn['FILTER']==band.upper()) &
                          (trest>-3) & (trest<1))[0]
            if len(ib) == 0:
                continue
            m = sn['MAG'][ib]
            merr = sn['MAGERR'][ib]
            t = trest[ib]

        # plot all inferred colors in the bottom panel
        for band1, color, marker in zip(
                ['f435w', 'f606w', 'f814w',
                 'f105w', 'f125w', 'f140w', 'f160w'],
                ['c', 'b', 'darkgreen', 'darkorange', 'r', 'm', 'darkred'],
                ['^', '>', 'v', 's', 'd', 'h', 'o']):
            ib = np.where((sn['FILTER']==band1.upper()) &
                          (trest>-3) & (trest<1))[0]
            if len(ib) == 0:
                continue
            m1 = sn['MAG'][ib]
            merr1 = sn['MAGERR'][ib]
            t1 = trest[ib]
            ax1.errorbar(t1, m1, merr1, marker=marker, ls= ' ', color=color,
                         ecolor='k', elinewidth=0.5, capsize=1., mew=0.5,
                         label=band1.upper())

            for band2, fillcolor in zip(['f814w', 'f125w', 'f160w'],
                                        ['darkgreen', 'r', 'darkred']):
                if band1==band2: continue
                if band1 in ['f140w','f160w'] and band2=='f125w': continue
                if band2 not in maginterpdict: continue
                if band2=='f160w':
                    mec=color
                    color='w'
                else:
                    mec='k'
                colorname = band1.upper() + '-' + band2.upper()
                slope, intercept = maginterpdict[band2][0]
                covmatrix = maginterpdict[band2][1]
                slope_err, intercept_err = np.sqrt(np.diagonal(covmatrix))
                slope_intercept_cov = covmatrix[0, 1]
                fiterrfunc = lambda x: np.sqrt((slope_err * x)**2 +
                                               intercept_err**2 +
                                               (2 * x * slope_intercept_cov))

                top = line(tplot, slope, intercept) + fiterrfunc(tplot)
                bot = line(tplot, slope, intercept) - fiterrfunc(tplot)

                ax1.fill_between(tplot, bot, top, color=fillcolor,
                                 alpha=0.1, zorder=-200)
                ax1.plot(tplot, line(tplot, slope, intercept),
                         marker=' ', ls='-', color=fillcolor,
                         label='__nolabel__', zorder=-100)

                m2 = line(t1, slope, intercept)
                merr2 = fiterrfunc(t1)
                c12 = m1 - m2
                c12err = np.sqrt(merr1**2 + merr2**2)

                ax2.errorbar(t1, c12, c12err,
                             marker=marker, ls= ' ', color=color, mec=mec,
                             ecolor='k', elinewidth=0.5, capsize=1., mew=0.5,
                             label=colorname)
                colorset += c12.tolist()
                colorerrset += c12err.tolist()

        meancolor = np.average(np.array(colorset),
                               weights=1/np.array(colorerrset)**2)
        if event=='se':
            ax1.legend(loc='lower right', fontsize='small', frameon=False,
                       ncol=1)
            pl.setp(ax2.get_yticklabels(), visible=False)
            ax2.legend(loc='lower right', fontsize='small', frameon=False,
                       ncol=2)
        else:
            ax1.legend(loc='lower right', fontsize='small', frameon=False)
            ax1.set_ylabel('AB Magnitude', labelpad=2)
            ax2.set_ylabel('Color', labelpad=-5)
            ax2.legend(loc='upper left', fontsize='small', frameon=False,
                       ncol=1, borderpad=1.8)

        ax2.axhline(meancolor, ls='--', lw=0.5, zorder=-1000)
        ax2.text(1.7, meancolor+0.05, '%.1f' % np.abs(np.round(meancolor,1)),
                 fontsize='small', ha='right', va='bottom')
        ax2.set_xlabel('t$_{\\rm rest}$ (days)', labelpad=5)
        ax1.set_ylim(30.15, 26.2)
        ax2.set_ylim(-0.9,2.48)
        ax1.set_xlim(-2.9,1.9)
        ax2.set_xlim(-2.9,1.9)
        pl.setp(ax1.get_xticklabels(), visible=False)
        if event=='se':
            pl.setp(ax1.get_yticklabels(), visible=False)

    fig = pl.gcf()
    fig.subplots_adjust(wspace=0, hspace=0, left=0.1, bottom=0.12,
                        right=0.97, top=0.97)

    return maginterpdict


def plot_colorcurve_binned( binsize=0.5 ):
    """ make a plot showing the color curves of spockSE and NW"""
    import sncosmo
    from astropy.io import ascii
    from matplotlib import ticker
    from pytools import plotsetup
    from astropy.table import Table,Column

    plotsetup.paperfig()

    sncosmo.plotting._cmap_wavelims = (4350, 15300)
    # read in the data
    nw = ascii.read('data/HST_FFSN_spockNW_phot.data', format='commented_header',header_start=-1, data_start=0 )
    se = ascii.read('data/HST_FFSN_spockSE_phot.data', format='commented_header',header_start=-1, data_start=0 )

    colorcurvedict = {
        'mjd':[],'mjderr':[],'colorname':[],'color':[],'colorerr':[] }


    for iax,src in zip([1,2],[nw, se]) :
        if iax==1 :
            xlim=[56669,56675]
            ylim = [-0.5,0.6]

        if iax==2:
            xlim = [56898,56902]
            ylim = [0,3]

        ax = pl.gcf().add_subplot( 2, 1, iax )
        if iax == 1 :
            ax.text( 0.1,0.9, 'Spock-NW', ha='left', va='top', fontsize='large', transform=ax.transAxes )
        if iax == 2 :
            ax.text( 0.1,0.9, 'Spock-SE', ha='left', va='top', fontsize='large', transform=ax.transAxes )
        mjd, mag, magerr = src['MJD'], src['MAG'], src['MAGERR']
        bandname, flux, fluxerr = src['FILTER'], src['FLUX'], src['FLUXERR']
        for thismjd in np.arange(xlim[0], xlim[1], binsize):
            thisbindict = {}
            for thisband in np.unique(bandname):
                ithisbinband = np.where((bandname == thisband) &
                                        (thismjd <= mjd) &
                                        (mjd < thismjd + binsize))[0]
                if len(ithisbinband) < 1:
                    continue
                thisflux, thisfluxerr = weighted_avg_and_std(
                    flux[ithisbinband], 1/fluxerr[ithisbinband]**2)
                thismag, thismagerr = weighted_avg_and_std(
                    mag[ithisbinband], 1/magerr[ithisbinband]**2)
                bandpass = sncosmo.get_bandpass(thisband.lower())
                thisbindict[thisband] = [bandpass.wave_eff,
                                         thisflux, thisfluxerr,
                                         thismag, thismagerr]
            #if 56898.9 < thismjd < 56900.1 :
            #    import pdb; pdb.set_trace()
            for key1, val1 in thisbindict.iteritems():
                for key2, val2 in thisbindict.iteritems():
                    if key1 == key2:
                        continue
                    if val1[0] >= val2[0]:
                        continue

                    bandpair = [key1, key2]
                    magpair = [val1[3], val2[3]]
                    magerrpair = [val1[4], val2[4]]
                    thismjdmid = thismjd + binsize/2.

                    iblue = np.argmin([val1[0],val2[0]])
                    ired = np.argmax([val1[0],val2[0]])

                    colorcurvedict['colorname'].append(bandpair[iblue].upper() + '-' + bandpair[ired].upper())
                    colorcurvedict['color'].append( magpair[iblue] - magpair[ired] )
                    colorcurvedict['colorerr'].append( np.sqrt(magerrpair[iblue]**2 + magerrpair[ired]**2 ) )
                    colorcurvedict['mjd'].append( thismjdmid )
                    colorcurvedict['mjderr'].append( binsize/2. )
        colortable = Table( colorcurvedict )

        for colorname in np.unique( colortable['colorname'] ):
            icolor = np.where(colortable['colorname'] == colorname)

            mjd = colortable['mjd'][icolor]
            color = colortable['color'][icolor]
            colorerr = colortable['colorerr'][icolor]
            mjderr = colortable['mjderr'][icolor]
            if min(mjd) > xlim[1] : continue
            if max(mjd) < xlim[0] : continue
            ax.errorbar( mjd,  color, colorerr, mjderr, marker='o', ls=' ', capsize=1, label=colorname )
        ax.set_xlabel('MJD')
        ax.set_ylabel('color (AB mag)')
        ax.legend( numpoints=1 )

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        ax.xaxis.set_major_locator( ticker.MultipleLocator( 1 ) )
        ax.xaxis.set_minor_locator( ticker.MultipleLocator( 0.2 ) )


    colortable.write( 'spock_colors.data', format='ascii.fixed_width' )
    pl.draw()
    return( colortable )


def plot_colorcurve_movingbin( binsize=0.5 ):
    """ make a plot showing the color curves of spockSE and NW"""
    import sncosmo
    from astropy.io import ascii
    from matplotlib import ticker
    from pytools import plotsetup
    from astropy.table import Table,Column

    plotsetup.paperfig()

    sncosmo.plotting._cmap_wavelims = (4350, 15300)
    # read in the data
    nw = ascii.read('data/HST_FFSN_spockNW_phot.data', format='commented_header',header_start=-1, data_start=0 )
    se = ascii.read('data/HST_FFSN_spockSE_phot.data', format='commented_header',header_start=-1, data_start=0 )

    colorcurvedict = {
        'mjd':[],'mjderr':[],'colorname':[],'color':[],'colorerr':[] }


    for iax,src in zip([1,2],[nw, se]) :
        if iax==1 :
            xlim=[56670,56674]
            ylim = [-0.5,0.6]

        if iax==2:
            xlim = [56898.5,56902]
            ylim = [0,3]

        ax = pl.gcf().add_subplot( 2, 1, iax )
        if iax == 1 :
            ax.text( 0.1,0.9, 'Spock-NW', ha='left', va='top', fontsize='large', transform=ax.transAxes )
        if iax == 2 :
            ax.text( 0.1,0.9, 'Spock-SE', ha='left', va='top', fontsize='large', transform=ax.transAxes )
        mjd, mag, magerr = src['MJD'], src['MAG'], src['MAGERR']
        bandname, flux, fluxerr = src['FILTER'], src['FLUX'], src['FLUXERR']
        # for each distinct observation...
        for i in range(len(mjd)):
            if flux[i] / fluxerr[i] < 3: continue
            thismjd = mjd[i]
            # find all other observations taken within the given MJD bin range
            # ithisbin = np.where((mjd-binsize/2.<=mjd) & (mjd<mjd+binsize/2.))[0]
            #if len(ithisbin)==1 : continue
            #thisband = bandname[ithisbin]
            #thisflux = flux[ithisbin]
            #thisfluxerr = fluxerr[ithisbin]
            #thismag = mag[ithisbin]
            #thismagerr = magerr[ithisbin]

            thisbindict = {}
            for thisband in np.unique(bandname):
                ithisbinband = np.where((bandname == thisband) &
                                        (thismjd - binsize / 2. <= mjd) &
                                        (mjd < thismjd + binsize / 2.) &
                                        (magerr > 0) & (magerr < 0.33))[0]
                if len(ithisbinband) < 1:
                    continue
                thisflux, thisfluxerr = weighted_avg_and_std(
                    flux[ithisbinband], 1/fluxerr[ithisbinband]**2)
                thismag, thismagerr = weighted_avg_and_std(
                    mag[ithisbinband], 1/magerr[ithisbinband]**2)
                bandpass = sncosmo.get_bandpass(thisband.lower())
                thisbindict[thisband] = [bandpass.wave_eff,
                                         thisflux, thisfluxerr,
                                         thismag, thismagerr]
                if thismagerr>0.5:
                    import pdb; pdb.set_trace()
            for key1, val1 in thisbindict.iteritems():
                for key2, val2 in thisbindict.iteritems():
                    if key1 == key2:
                        continue
                    if val1[0] >= val2[0]:
                        continue

                    bandpair = [key1, key2]
                    magpair = [val1[3], val2[3]]
                    magerrpair = [val1[4], val2[4]]
                    thismjdmid = thismjd + binsize/2.

                    iblue = np.argmin([val1[0],val2[0]])
                    ired = np.argmax([val1[0],val2[0]])

                    colorcurvedict['colorname'].append(bandpair[iblue].upper() + '-' + bandpair[ired].upper())
                    colorcurvedict['color'].append( magpair[iblue] - magpair[ired] )
                    colorcurvedict['colorerr'].append( np.sqrt(magerrpair[iblue]**2 + magerrpair[ired]**2 ) )
                    colorcurvedict['mjd'].append( thismjdmid )
                    colorcurvedict['mjderr'].append( binsize/2. )
        colortable = Table( colorcurvedict )

        for colorname in np.unique( colortable['colorname'] ):
            icolor = np.where(colortable['colorname'] == colorname)

            mjd = colortable['mjd'][icolor]
            color = colortable['color'][icolor]
            colorerr = colortable['colorerr'][icolor]
            mjderr = colortable['mjderr'][icolor]
            if min(mjd) > xlim[1] : continue
            if max(mjd) < xlim[0] : continue
            ax.errorbar( mjd,  color, colorerr, mjderr, marker='o', ls=' ', capsize=1, label=colorname )
        ax.set_xlabel('MJD')
        ax.set_ylabel('color (AB mag)')
        ax.legend( numpoints=1 )

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        ax.xaxis.set_major_locator( ticker.MultipleLocator( 1 ) )
        ax.xaxis.set_minor_locator( ticker.MultipleLocator( 0.2 ) )


    colortable.write( 'spock_colors.data', format='ascii.fixed_width' )
    pl.draw()
    return( colortable )

