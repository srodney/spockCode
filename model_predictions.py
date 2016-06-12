from astropy.io import ascii
from astropy.table import Table
import numpy as np
import os
import sys
import corner

thisfile = sys.argv[0]
if 'ipython' in thisfile : thisfile = __file__
thispath = os.path.abspath( os.path.dirname( thisfile ) )



def mk_corner_plot(model='oguri', nbins=None):

    if model == 'oguri':
        datfile = os.path.join(thispath,'data/oguri.dat')
        dat = ascii.read(datfile, format='commented_header',
                         data_start=0, header_start=-1)

        # compute new columns from the data for plotting
        dat['mu1'] = np.abs(dat['mu1'])  # absolute value of magnifications
        dat['dt13'] = - dat['dt31']  # time delays relative to the NW event
        dat['dt12'] = dat['dt31'] - dat['dt32']
        modelrange = [(-10,90),(-10,250),(2,4),(-4,18),0.999]
        weights = None
        if nbins is None:
            nbins = 20

    elif model == 'zitrin':
        datfile = os.path.join(thispath,'data/zitrin.dat')
        dat = ascii.read(datfile, format='commented_header',
                         data_start=0, header_start=-1)
        modelrange = [(20,220),(10,70),(2.8,4.5),(15,70), 0.999]
        weights = None
        if nbins is None:
            nbins = 15

    elif model == 'jauzac1':
        datfile = os.path.join(thispath,'data/m0416_timedel-mag_bfit.fits')
        dat = Table.read(datfile)
        dat['MAGS1'].name = 'mu1'
        dat['MAGS2'].name = 'mu2'
        dat['MAGS3'].name = 'mu3'
        dat['EMAGS1'].name = 'errmu1'
        dat['EMAGS2'].name = 'errmu2'
        dat['DTD_S2'].name = 'dt12'
        dat['DTD_S3'].name = 'dt13'
        weights = 1 / np.sqrt(dat['errmu1']**2 + dat['errmu2']**2)
        modelrange = [0.999, 0.999, 0.999, 0.999, 0.999]
        if nbins is None:
            nbins = 15

    elif model == 'jauzac2':
        datfile = os.path.join(thispath,'data/m0416_spock_timedel-mag_bfit.fits')
        dat = Table.read(datfile)
        dat['MAGS1'].name = 'mu1'
        dat['MAGS2'].name = 'mu2'
        dat['MAGS3'].name = 'mu3'
        dat['EMAGS1'].name = 'errmu1'
        dat['EMAGS2'].name = 'errmu2'
        dat['DTD_S2'].name = 'dt12'
        dat['DTD_S3'].name = 'dt13'
        weights = 1 / np.sqrt(dat['errmu1']**2 + dat['errmu2']**2)
        modelrange = [0.999, 0.999, 0.999, 0.999, 0.999]
        if nbins is None:
            nbins = 15

    elif model == 'williams':
        datfile = os.path.join(thispath, 'data/williams.dat')
        dat = ascii.read(datfile, format='commented_header',
                         data_start=0, header_start=-1)
        dat['dt12'] = -dat['dt21']
        dat['dt13'] = -dat['dt31']
        dat['mu1'] = np.abs(dat['mu1'])
        weights = 1/ np.sqrt(dat['delx1']**2 + dat['dely1']**2 +
                             dat['delx2']**2 + dat['dely2']**2)
        modelrange = [(-2,25), (-2,35), (0,5), (-50,10), (-4500, -300)]
        if nbins is None:
            nbins = 12


    # reformat as an ordered array of samples for corner plotting
    newtable = Table([dat['mu1'], dat['mu2'], dat['mu3'],
                      dat['dt12'], dat['dt13']])
    newdat = np.array([list(d) for d in newtable.as_array()])

    labels = ['$\mu_{\\rm NW}$', '$\mu_{\\rm SE}$', '$\mu_{\\rm 11.3}$',
              '$\Delta t_{\\rm NW:SE}$','$\Delta t_{\\rm NW:11.3}$',
              ]

    corner.corner(newdat, bins=nbins, range=modelrange,
                  weights=weights, color=u'k',
                  smooth=None, smooth1d=None,
                  labels=labels, label_kwargs={'fontsize':'large'},
                  show_titles=False, title_fmt=u'.2f', title_kwargs=None,
                  truths=None, truth_color=u'#4682b4',
                  scale_hist=False, quantiles=[0.16,0.84],
                  verbose=False, fig=None, max_n_ticks=5,
                  top_ticks=False, use_math_text=False, hist_kwargs=None)

    # print out the mean and standard deviation from the corner plots
    modelresults = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                     zip(*np.percentile(newdat, [16, 50, 84], axis=0)))
    for label, modres in zip(labels, modelresults):
        print '%s = %.2f +%.2f -%.2f' % (
            label, modres[0], modres[1], modres[2])

    return dat




