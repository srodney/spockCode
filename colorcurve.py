from __future__ import print_function
from constants import  __THISDIR__, __ABRESTBANDNAME__, __VEGARESTBANDNAME__
from constants import __MJDPKNW__, __MJDPKSE__, __Z__
from constants import __MJDPREPK0NW__, __MJDPOSTPK0NW__
from constants import __MJDPREPK0SE__, __MJDPOSTPK0SE__
from . import lightcurve
from .kcorrections import compute_kcorrection, get_linfitmag, get_kcorrection

# from scipy import interpolate as scint
from scipy import optimize as scopt
import numpy as np
from matplotlib import pyplot as pl
from pytools import plotsetup
from matplotlib import rcParams
import os
from astropy.io import ascii
import sncosmo


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

        # plot observer-frame measured mags in the top panel, with linear
        # fit lines overlaid, then plot interpolated observer-frame colors in
        # the bottom panel
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
                colorname = band1.upper() + '-' + band2.upper()

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


def compute_kcorrected_colors_from_observed_data():
    """ calculate the observed colors in rest-frame bands,
     applying K corrections.
    :param restphotsys:
    :return:
    """
    # read in the observed spock data
    nw, se = lightcurve.get_spock_data()

    # read in the data file produced by linear fits to the pre-peak data
    indatfile = os.path.join(__THISDIR__, 'data/magpk_trise_tfall.dat')
    fitdat = ascii.read(indatfile, format='commented_header', header_start=-1,
                        data_start=0)

    outfile = os.path.join(__THISDIR__, 'data/observed_colors_kcorrected.dat')
    fout = open(outfile,'w')
    print("# event  trest c1abname c1ab c2abname c2ab  "
          "c1veganame c1vega c2veganame c2vega ", file=fout)

    # for each observed data point from -6 days to 0 days in the observer
    # frame, get the interpolated magnitudes from the linear fits
    # and the observed magnitudes from the light curve data
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
        # NOTE: the rest-frame time is always defined relative to the
        #  *observed* MJD of peak brightness, not the assumed mjdpk
        mjd = sn['MJD']
        trest = (mjd-mjdpkobs)/(1+__Z__)
        tprepk0 = (mjdprepk0-mjdpkobs)/(1+__Z__)
        tpostpk0 = (mjdpostpk0-mjdpkobs)/(1+__Z__)

        trestfit = fitdat['deltatpk']
        mabfit = fitdat['mpk']
        fitfilterlist = np.unique(fitdat['band'])
        inearpeak = np.where((trest>tprepk0) & (trest<=0))[0]
        for i in inearpeak:
            # for each observed data point,
            # construct a crude SED from the linear fits
            # and this observed data point
            obsbandname = sn['FILTER'][i].lower()
            #if obsbandname in fitfilterlist:
            #    continue

            source_wave = []
            source_flux = []

            trest = (sn['MJD'][i] - mjdpkobs)/(1+__Z__)
            if trest<np.min(trestfit): continue
            if trest>np.max(trestfit): continue
            bandpass = sncosmo.get_bandpass(obsbandname)
            source_wave.append(bandpass.wave_eff)
            source_flux.append(sn['MAG'][i])

            ifit = np.where((np.abs(trestfit - trest) < 0.1) &
                            (fitdat['event'] == event))[0]
            for j in ifit:
                bandpass = sncosmo.get_bandpass(fitdat['band'][j])
                source_wave.append(bandpass.wave_eff)
                source_flux.append(fitdat['mpk'][j])

            isorted = np.argsort(source_wave)

            source_wave = np.array(source_wave)
            source_flux = np.array(source_flux)
            abrestbandname = __ABRESTBANDNAME__[obsbandname]
            vegarestbandname = __VEGARESTBANDNAME__[obsbandname]
            abkcor = compute_kcorrection(abrestbandname, obsbandname,
                                         __Z__, source_wave[isorted],
                                         source_flux[isorted],
                                         source_wave_unit='Angstrom',
                                         source_flux_unit='magab',
                                         obsphotsys='AB', restphotsys='AB',
                                         verbose=False)

            vegakcor = compute_kcorrection(vegarestbandname, obsbandname,
                                         __Z__, source_wave[isorted],
                                         source_flux[isorted],
                                         source_wave_unit='Angstrom',
                                         source_flux_unit='magab',
                                         obsphotsys='AB', restphotsys='AB',
                                         verbose=False)

            # To construct a color measurement, we also need the interpolated
            # magnitude from a redder band. We get these from the linear fits
            # made to well sampled bands, reading in from the data file that
            # was produced by the peak_luminosity_vs_time.py module
            if obsbandname.lower().startswith('f1'):
                fitbandname1 = 'f125w'
                fitbandname2 = 'f160w'
            else:
                fitbandname1 = 'f435w'
                fitbandname2 = 'f814w'
            m1fit = get_linfitmag(event, fitbandname1, trest)
            m2fit = get_linfitmag(event, fitbandname2, trest)

            # now we get the K corrections to convert from the observer-frame
            # band to the rest-frame band and the AB or Vega system
            kcor1vega = get_kcorrection(event, fitbandname1, trest, restphotsys='Vega')
            kcor2vega = get_kcorrection(event, fitbandname1, trest, restphotsys='Vega')
            kcor1ab = get_kcorrection(event, fitbandname1, trest, restphotsys='AB')
            kcor2ab = get_kcorrection(event, fitbandname1, trest, restphotsys='AB')

            fitbandname1restAB = __ABRESTBANDNAME__[fitbandname1]
            fitbandname2restAB = __ABRESTBANDNAME__[fitbandname2]

            fitbandname1restVega = __VEGARESTBANDNAME__[fitbandname1]
            fitbandname2restVega = __VEGARESTBANDNAME__[fitbandname2]


            # Here is the observed magnitude in the bluer band
            mobs = sn['MAG'][i]

            # now we can compute the AB or Vega color in rest-frame band passes
            cab1 = (mobs + abkcor) - (m1fit + kcor1ab)
            cab2 = (mobs + abkcor) - (m2fit + kcor2ab)
            cvega1 = (mobs + vegakcor) - (m1fit + kcor1vega)
            cvega2 = (mobs + vegakcor) - (m2fit + kcor2vega)

            obscolorname1 = '%s-%s' % (obsbandname.lower(), fitbandname1)
            obscolorname2 = '%s-%s' % (obsbandname.lower(), fitbandname2)
            abcolorname1 = '%s-%s'%(abrestbandname[4:], fitbandname1restAB[4:])
            abcolorname2 = '%s-%s'%(abrestbandname[4:], fitbandname2restAB[4:])
            vegacolorname1 = '%s-%s'%(vegarestbandname[7:], fitbandname1restVega[7:])
            vegacolorname2 = '%s-%s'%(vegarestbandname[7:], fitbandname2restVega[7:])

            print("%s %.1f  %4s  %6.1f  %4s  %6.1f  %4s  %6.1f  %4s  %6.1f  " % (
                event, trest, abcolorname1, cab1, abcolorname2, cab2,
                vegacolorname1, cvega1, vegacolorname2, cvega2), file=fout)
    fout.close()


