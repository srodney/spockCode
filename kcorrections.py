from __future__ import print_function
from constants import (__Z__, __ABRESTBANDNAME__, __VEGARESTBANDNAME__,
                       __THISDIR__)

from scipy import interpolate as scint
from scipy import integrate as scintegrate
import numpy as np
from astropy.io import ascii
import sncosmo
import os



def compute_kcorrections(restphotsys='Vega'):
    """ Derive K corrections for the Spock events at times preceding the
    observed date of peak brightness, and extending to times beyond the
    observed peak (assuming the light curve continued to rise linearly in mag
    space).  Each K correction is based on linear fits to the light curves
    in a few well-sampled bands.

    With restphotsys='Vega', the K corrections are for translating from
    measured photometry in AB mags through HST bandpasses into rest-frame
    photometry in Vega mags through Bessell UBVRI bands.
    With restphotsys='AB', the K corrections convert from HST bands and AB mags
    into SDSS ugriz bands, also in AB mags.
    """
    # read in the data file produced by linear fits to the pre-peak data
    indatfile = os.path.join(__THISDIR__, 'data/magpk_trise_tfall.dat')
    indat = ascii.read(indatfile, format='commented_header', header_start=-1,
                       data_start=0)
    outdatfile = os.path.join(
        __THISDIR__, 'data/magpk_trise_tfall_kcor_%s.dat'%restphotsys.lower())
    fout = open( outdatfile, 'w')
    print('# event obsband deltatpk mpk fnupk trise t2 t3 restband kcor',
          file=fout)

    # Define the spectral density of flux for the 'standard' zero-magnitude
    # source in the emitted-frame (Q) and the observed frame (R).
    # Also select the correct __RESTBANDNAME__ dict which holds the manually
    # identified rest-frame band that most closely matches each
    # obs-frame filter.
    if restphotsys=='Vega':
        vega = sncosmo.get_magsystem('vega')
        vegainterp = scint.interp1d(
            vega._refspectrum.wave, vega._refspectrum.flux,
            bounds_error=False, fill_value=0, assume_sorted=True)
        # Flux density from sncosmo is Flambda, in units
        # of [erg / (Angstrom cm2 sec)].  We want Fnu, in units of
        # microJanskys, for consistency with the observed flux measurements
        # This conversion factor includes the factor of 1/c (with c in
        # Angstrom / sec) to get from Flambda to Fnu, and the conversion
        # from [erg/(cm2 sec Hz)] to microJanskys.
        conversion_factor = 0.33356409519815206 * 1e14
        gnuQ = lambda l: vegainterp(l) * l * l * conversion_factor
        __RESTBANDNAME__ = __VEGARESTBANDNAME__
    else:
        # For K corrections to the AB system we have the simpler AB standard
        # source that is flat in Fnu. We make this into a python lambda
        # function anyway, to match the Vega version.
        gnuQ = lambda l: 3631 * 1e9 # microJanskys
        __RESTBANDNAME__ = __ABRESTBANDNAME__

    # observed magnitudes are on the AB system
    gnuR = 3631 * 1e9 # microJanskys

    for event in ['nw','se']:
        # get a list of the observed bandpasses for this event
        ievent = np.where(indat['event'] == event)[0]
        obsbandnamelist = np.array(np.unique(indat['band'][ievent]), dtype=str)

        # define dictionaries to hold sncosmo Bandpass objects for each
        # observer-frame band and the corresponding rest-frame band.
        obsbandpassdict = dict(
            [(obsbandname, sncosmo.get_bandpass(obsbandname))
             for obsbandname in np.unique(indat['band'])])
        restbandpassdict = dict(
            [(obsbandname,sncosmo.get_bandpass(__RESTBANDNAME__[obsbandname]))
             for obsbandname in np.unique(indat['band'])])

        # fill in the dictionaries with the bandpass transmission functions
        obsbandtransdict = {}
        restbandtransdict = {}
        for obsbandname in obsbandnamelist:
            obsbandpass = obsbandpassdict[obsbandname]
            obsbandtransdict[obsbandname] = scint.interp1d(
                obsbandpass.wave, obsbandpass.trans, bounds_error=False,
                fill_value=0)
            restbandpass = restbandpassdict[obsbandname]
            restbandtransdict[obsbandname] = scint.interp1d(
                restbandpass.wave, restbandpass.trans, bounds_error=False,
                fill_value=0)

        """ For each time prior to peak or each assumed time of peak
        brightness, we will define a crude
        SED as a linear interpolation in flux (fnu) vs wavelength space
        between the two "well-sampled" bands for this event.
        Note: We are working with fnu in wavelength space, but this is
        ok b/c the factors of c/wave^2 that translate fnu to flambda are
        handled in the definitions of the K-corr integrands below. """
        dtpklist = list(np.unique(indat['deltatpk'][ievent]))
        for trelpk in dtpklist:
            # First, collect fnu and central wavelength values for each band
            fnulist = []
            wavelist = []
            for obsbandname in obsbandnamelist:
                i = np.where((indat['event'] == event) &
                             (indat['band'] == obsbandname) &
                             (indat['deltatpk'] == trelpk))[0]
                if len(i)<1 : continue
                obsbandpass = obsbandpassdict[obsbandname]
                wave_eff = obsbandpass.wave_eff  # in Angstroms
                mpk = float(indat['mpk'][i])
                fnu = 10**((23.9-mpk)/2.5) # flux density in microJanskys
                wavelist.append(wave_eff)
                fnulist.append(fnu)

            # define a piecewise linear interpolation through the given bands
            def fnuinterp(w):
                return (np.where(
                    w>=max(wavelist),
                    scint.interp1d(wavelist, fnulist, kind='linear',
                                   bounds_error=False,
                                   fill_value=fnulist[np.argmax(wavelist)]
                                   )(w),
                    scint.interp1d(wavelist, fnulist, kind='linear',
                                   bounds_error=False,
                                   fill_value=fnulist[np.argmin(wavelist)]
                                   )(w)))

            for obsbandname in obsbandnamelist:
                irow = np.where((indat['event'] == event) &
                                (indat['deltatpk'] == trelpk) &
                                (indat['band'] == obsbandname))[0]
                row = indat[irow]
                # define the K correction integrands (following Hogg on NED)
                # and integrate them numerically
                obsbandpass = obsbandpassdict[obsbandname]
                obsbandtrans = obsbandtransdict[obsbandname]
                restbandname = __RESTBANDNAME__[obsbandname]
                restbandpass = restbandpassdict[obsbandname]
                restbandtrans = restbandtransdict[obsbandname]

                def obs_source_integrand(w):
                    return (fnuinterp(w)/w) * obsbandtrans(w)
                obs_source_integrated = scintegrate.trapz(
                    obs_source_integrand(obsbandpass.wave), obsbandpass.wave)
                def obs_band_integrand(w):
                    return (1/w) * gnuR * obsbandtrans(w)
                obs_band_integrated = scintegrate.trapz(
                    obs_band_integrand(obsbandpass.wave), obsbandpass.wave)
                def rest_band_integrand(w):
                    return (1/w) * gnuQ(w) * restbandtrans(w)
                rest_band_integrated = scintegrate.trapz(
                    rest_band_integrand(restbandpass.wave), restbandpass.wave)
                def rest_source_integrand(w):
                    return (fnuinterp((1+__Z__)*w)/w) * restbandtrans(w)
                rest_source_integrated = scintegrate.trapz(
                    rest_source_integrand(restbandpass.wave),
                    restbandpass.wave)

                kcorval = -2.5 * np.log10(
                        (1 + __Z__) *
                        (obs_source_integrated * rest_band_integrated) /
                        (obs_band_integrated * rest_source_integrated))

                rowstr = str(row).split('\n')[-1]
                print("%s  %15s  %.3f " % ( rowstr, restbandname, kcorval ),
                      file=fout)
    fout.close()


from matplotlib import pyplot as pl
def plot_kcorrections():
    """
    :param restphotsys: rest-frame photometric system
    :return:
    """

    pl.clf()
    fig = pl.gcf()
    ax1 = fig.add_subplot(1,2,1)
    ax2 = fig.add_subplot(1,2,2, sharex=ax1, sharey=ax1)
    for ax, restphotsys in zip([ax1,ax2], ['Vega', 'AB']):
        kcordatfile = os.path.join(
            __THISDIR__,
            "data/magpk_trise_tfall_kcor_%s.dat" % restphotsys.lower())
        data = ascii.read(kcordatfile, format='commented_header',
                          header_start=-1, data_start=0)
        if restphotsys.lower() == 'vega':
            __RESTBANDNAME__ = __VEGARESTBANDNAME__
        else:
            __RESTBANDNAME__ = __ABRESTBANDNAME__
        obsbandlist = np.unique(data['obsband'])

        for obsband in obsbandlist:
            iband = np.where(data['obsband'] == obsband)[0]
            restband = __RESTBANDNAME__[obsband.lower()]
            ax.plot(data['deltatpk'][iband], data['kcor'][iband],
                    ls='-', marker=' ',
                    label='%s to %s  (%s)' % (
                        obsband.upper(), restband.lower(), restphotsys))
        ax.legend()
    fig.text(0.5, 0.01, 'rest-frame time, relative to observed peak',
             transform=fig.transFigure, ha='center', va='bottom')
    ax1.set_ylabel('K correction (mag)')

    fig.subplots_adjust(left=0.08, right=0.95, bottom=0.12, top=0.97)

