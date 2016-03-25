from constants import __MJDPKNW__, __MJDPKSE__, __Z__
from constants import __MJDPREPK0NW__, __MJDPOSTPK0NW__
from constants import __MJDPREPK0SE__, __MJDPOSTPK0SE__
from . import lightcurve
from scipy import interpolate as scint
from scipy import integrate as scintegrate
import numpy as np
from matplotlib import pyplot as pl
from pytools import plotsetup
from astropy.io import ascii
import sncosmo


def compute_kcorrections():
    # read in the data file produced by linear fits to the pre-peak data
    indatfile = 'data/magpk_trise_tfall.dat'
    indat = ascii.read(indatfile, format='commented_header', header_start=-1,
                     data_start=0)
    fout = open('data/kcor_vs_time.dat', 'w')
    print >> fout, '# event  obsband  restband   deltatrest   kcor'

    absys = sncosmo.get_magsystem('ab')

    # manually identified the rest-frame band that most closely matches each
    # obs-frame filter.
    restbandname = {'f435w':'bessellux','f814w':'bessellb',
                'f125w':'sdssr', 'f160w':'sdssi'}
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
            [(obsbandname,sncosmo.get_bandpass(restbandname[obsbandname]))
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

        """ For each assumed time of peak brightness, we will define a crude
        SED as a linear interpolation in flux (fnu) vs wavelength space
        between the two "well-sampled" bands for this event.
        Note: We are working with fnu in wavelength space, but this is
        ok b/c the factors of c/wave^2 that translate fnu to flambda are
        handled in the definitions of the K-corr integrands below. """
        for dtpk in np.unique(indat['deltatpk'][ievent]):
            # First, collect fnu and central wavelength values for each band
            fnulist = []
            wavelist = []
            for obsbandname in obsbandnamelist:
                i = np.where((indat['event'] == event) &
                             (indat['band'] == obsbandname) &
                             (indat['deltatpk'] == dtpk))[0]
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

            # define the K correction integrands (following Hogg on NED)
            # and integrate them numerically
            for obsbandname in obsbandnamelist:
                obsbandpass = obsbandpassdict[obsbandname]
                obsbandtrans = obsbandtransdict[obsbandname]
                def obs_source_integrand(w):
                    return (fnuinterp(w)/w) * obsbandtrans(w)
                obs_source_integrated = scintegrate.trapz(
                    obs_source_integrand(obsbandpass.wave), obsbandpass.wave)
                def obs_band_integrand(w):
                    return (1/w) * obsbandtrans(w)
                obs_band_integrated = scintegrate.trapz(
                        obs_band_integrand(obsbandpass.wave), obsbandpass.wave)

                restbandpass = restbandpassdict[obsbandname]
                restbandtrans = restbandtransdict[obsbandname]
                def rest_band_integrand(w):
                    return (1/w) * restbandtrans(w)
                rest_band_integrated = scintegrate.trapz(
                    rest_band_integrand(restbandpass.wave),
                    restbandpass.wave)
                def rest_source_integrand(w):
                    return (fnuinterp((1+__Z__)*w)/w) * restbandtrans(w)
                rest_source_integrated = scintegrate.trapz(
                    rest_source_integrand(restbandpass.wave),
                    restbandpass.wave)

                kcorval = -2.5 * np.log10(
                    (1 + __Z__) *
                    (obs_source_integrated * rest_band_integrated) /
                    (obs_band_integrated * rest_source_integrated))

                print >> fout, "%s  %s  %s  %.3e  %.3f" % (
                    event, obsbandname, restbandname[obsbandname],
                    dtpk, kcorval)
    fout.close()