from __future__ import print_function
from constants import  __THISDIR__, __ABRESTBANDNAME__, __VEGARESTBANDNAME__
from constants import __MJDPKNW__, __MJDPKSE__, __Z__
from constants import __MJDPREPK0NW__, __MJDPOSTPK0NW__
from constants import __MJDPREPK0SE__, __MJDPOSTPK0SE__

from . import lightcurve
from scipy import interpolate as scint
from scipy import integrate as scintegrate
import numpy as np
from astropy.io import ascii
import sncosmo
import os
from matplotlib import pyplot as pl


def compute_spock_kcorrections_from_linear_fits(restphotsys='Vega'):
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
        # get a list of the bandpasses for which we have a linear fit
        ievent = np.where(indat['event'] == event)[0]
        fitbandnamelist = np.array(np.unique(indat['band'][ievent]), dtype=str)

        # define dictionaries to hold sncosmo Bandpass objects for each
        # observer-frame band and the corresponding rest-frame band.
        obsbandpassdict = dict(
            [(obsbandname, sncosmo.get_bandpass(obsbandname))
             for obsbandname in fitbandnamelist])
        restbandpassdict = dict(
            [(obsbandname,sncosmo.get_bandpass(__RESTBANDNAME__[obsbandname]))
             for obsbandname in fitbandnamelist])

        # fill in the dictionaries with the bandpass transmission functions
        obsbandtransdict = {}
        restbandtransdict = {}
        for obsbandname in fitbandnamelist:
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
            for obsbandname in fitbandnamelist:
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
                return np.where(
                    w >= max(wavelist),
                    scint.interp1d(wavelist, fnulist, kind='linear',
                                   bounds_error=False,
                                   fill_value=fnulist[np.argmax(wavelist)]
                                   )(w),
                    scint.interp1d(wavelist, fnulist, kind='linear',
                                   bounds_error=False,
                                   fill_value=fnulist[np.argmin(wavelist)]
                                   )(w))

            for obsbandname in fitbandnamelist:
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

                irow = np.where((indat['event'] == event) &
                                (indat['deltatpk'] == trelpk) &
                                (indat['band'] == obsbandname))[0]
                row = indat[irow]
                rowstr = str(row).split('\n')[-1]
                print("%s  %15s  %.3f " % (rowstr, restbandname, kcorval),
                      file=fout)
    fout.close()


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

def get_kcorrection(event, obsband, trestrelpk, restphotsys='Vega'):
        kcordatfile = os.path.join(
            __THISDIR__,
            "data/magpk_trise_tfall_kcor_%s.dat" % restphotsys.lower())
        data = ascii.read(kcordatfile, format='commented_header',
                          header_start=-1, data_start=0)
        iobs = np.where((data['event']==event) &
                        (data['obsband']==obsband.lower()))[0]
        trelpk = data['deltatpk'][iobs]
        kcor = data['kcor'][iobs]
        interpolator = scint.interp1d(trelpk, kcor)
        return interpolator(trestrelpk)


def get_linfitmag(event, obsband, trestrelpk):
        linfitdatfile = os.path.join(__THISDIR__, "data/magpk_trise_tfall.dat")
        data = ascii.read(linfitdatfile, format='commented_header',
                          header_start=-1, data_start=0)
        iobs = np.where((data['event']==event) &
                        (data['band']==obsband.lower()))[0]
        trelpk = data['deltatpk'][iobs]
        maglinfit = data['mpk'][iobs]
        interpolator = scint.interp1d(trelpk, maglinfit)
        return interpolator(trestrelpk)


def plot_filter_curves():
    import sncosmo
    fig = pl.gcf()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    for band, col, name in zip(['ux','b','v','r','i'],
                               ['darkorchid','b','g','r','darkorange'],
                               ['U','B','V','R','I']):
        R = sncosmo.get_bandpass('bessell' + band)
        ax1.plot(R.wave * (1 + __Z__), R.trans, color=col, ls='-', label=name)

    for band, col, name in zip(['u','g','r','i'],
                               ['darkorchid','g','r','darkorange'],
                               ["u'","g'","r'","i'"]):
        R = sncosmo.get_bandpass('sdss' + band)
        ax2.plot(R.wave * (1 + __Z__), R.trans, color=col, ls='-', label=name)

    for band in ['f435w','f606w','f814w','f105w','f125w','f140w','f160w']:
        R = sncosmo.get_bandpass(band)
        ax1.plot(R.wave, R.trans, color='k', ls='--', label='__nolegend__')
        ax2.plot(R.wave, R.trans, color='k', ls='--', label='__nolegend__')

    ax1.legend()
    ax2.legend()
    pl.draw()


def compute_kcorrection(restbandname, obsbandname, redshift,
                        source_wave, source_flux,
                        source_wave_unit='Angstrom',
                        source_flux_unit='ABmag',
                        obsphotsys='AB', restphotsys='AB',
                        verbose=False):
    """ compute k corrections for a source at the given redshift
    :param restbandname: the name of an sncosmo bandpass (rest-frame)
    :param obsbandname: the name of an sncosmo bandpass (observed)
    :param redshift: redshift of the source
    :param source_wave: array of wavelength values for the source SED
    :param source_flux: array of flux or mag values for the source SED
    :param source_wave_unit: physical units for the source wavelength array
           choose from ['Angstrom','um']
    :param source_flux_unit: physical units for the source flux array
           choose from ['microJansky','magAB','erg/(Angstrom cm2 sec)']
    :return:
    """
    # Some lux densities, e.g. from sncosmo, are Flambda, in units
    # of [erg / (Angstrom cm2 sec)].  We want Fnu, in units of
    # microJanskys, for consistency with the observed flux measurements
    # This conversion factor includes the factor of 1/c (with c in
    # Angstrom / sec) to get from Flambda to Fnu, and the unit conversion
    # from [erg/(cm2 sec Hz)] to microJanskys.  Note that you also have
    # to include the factor of 1/wavelength**2 to convert from Flambda to Fnu
    flam2fnu_conversion_factor = 0.33356409519815206 * 1e14

    # get sncosmo Bandpass objects for the
    # observer-frame band and the corresponding rest-frame band.
    obsbandpass = sncosmo.get_bandpass(obsbandname)
    restbandpass = sncosmo.get_bandpass(restbandname)

    # get the bandpass transmission functions
    obsbandtrans = scint.interp1d(
        obsbandpass.wave, obsbandpass.trans, bounds_error=False,
        fill_value=0)
    restbandtrans = scint.interp1d(
        restbandpass.wave, restbandpass.trans, bounds_error=False,
        fill_value=0)

    # Define the spectral density of flux for the 'standard' zero-magnitude
    # source in the emitted-frame (Q) and the observed frame (R).
    # Also select the correct __RESTBANDNAME__ dict which holds the manually
    # identified rest-frame band that most closely matches each
    # obs-frame filter.
    if restphotsys.lower()=='vega' or obsphotsys.lower()=='vega':
        vega = sncosmo.get_magsystem('vega')
        vegainterp = scint.interp1d(
            vega._refspectrum.wave, vega._refspectrum.flux,
            bounds_error=False, fill_value=0, assume_sorted=True)
        gnuVega = lambda l: vegainterp(l) * l * l * flam2fnu_conversion_factor
    if restphotsys.lower()=='vega':
        gnuQ = gnuVega
    if obsphotsys.lower()=='vega':
        gnuR = gnuVega

    # For K corrections to/from the AB system we have the simpler AB standard
    # source that is flat in Fnu. We make this into a python lambda
    # function anyway, to match the Vega version.
    gnuAB = lambda l: 3631 * 1e9 # microJanskys
    if restphotsys.lower()=='ab':
        gnuQ = gnuAB
    if obsphotsys.lower()=='ab':
        gnuR = gnuAB

    if source_wave_unit.lower().startswith('angstrom'):
        source_wave_angstrom = source_wave
    elif source_wave_unit.lower().startswith('um'):
        source_wave_angstrom = 10000 * source_wave
    elif source_wave_unit.lower().startswith('nm'):
        source_wave_angstrom = 10 * source_wave
    else:
        raise RuntimeError("source wavelength unit %s not recognized" %
                           source_wave_unit)

    if source_flux_unit.lower().startswith('microjansky'):
        source_flux_microjanskys = source_flux
    elif source_flux_unit.lower() == 'magab':
        source_flux_microjanskys = 10**(23.9-source_flux)/2.5
    elif source_flux_unit.lower() == 'erg/(Angstrom cm2 sec)':
        source_flux_microjanskys = (source_flux * source_wave * source_wave *
                                    flam2fnu_conversion_factor)
    else:
        raise RuntimeError("source flux unit %s not recognized" %
                           source_flux_unit)

    # Define the K correction integrands (following K-corrections guide
    # from D. Hogg via NED) and integrate them numerically

    # Define an interpolation function to compute the flux in units of uJy
    fnuinterp = scint.interp1d(source_wave_angstrom, source_flux_microjanskys,
                               bounds_error=False, fill_value=0)

    wave_observed = np.arange(obsbandpass.wave.min(),
                              obsbandpass.wave.max(), 10)
    wave_emitted = np.arange(obsbandpass.wave.min()/(1+redshift),
                             obsbandpass.wave.max()/(1+redshift), 10)

    def obs_source_integrand(w):
        return (fnuinterp(w)/w) * obsbandtrans(w)
    obs_source_integrated = scintegrate.trapz(
        obs_source_integrand(wave_observed),
        wave_observed)
    def obs_band_integrand(w):
        return (1/w) * gnuR(w) * obsbandtrans(w)
    obs_band_integrated = scintegrate.trapz(
        obs_band_integrand(wave_observed),
        wave_observed)
    def rest_band_integrand(w):
        return (1/w) * gnuQ(w) * restbandtrans(w)
    rest_band_integrated = scintegrate.trapz(
        rest_band_integrand(wave_emitted),
        wave_emitted)
    def rest_source_integrand(w):
        return (fnuinterp((1+__Z__)*w)/w) * restbandtrans(w)
    rest_source_integrated = scintegrate.trapz(
        rest_source_integrand(wave_emitted),
        wave_emitted)

    kcorval = -2.5 * np.log10(
            (1 + __Z__) *
            (obs_source_integrated * rest_band_integrated) /
            (obs_band_integrated * rest_source_integrated))

    if verbose:
        print("%10s %10s %.3f" % (
            restbandname, obsbandname, kcorval))
    return kcorval


def compute_spock_kcorrections_from_observed_data(restphotsys='Vega'):
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

