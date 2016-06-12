from astropy.io import ascii
from astropy.table import Table
import numpy as np
import os
import sys
import corner

thisfile = sys.argv[0]
if 'ipython' in thisfile : thisfile = __file__
thispath = os.path.abspath( os.path.dirname( thisfile ) )


class LensModel(object):
    def __init__(self, modeler=None):
        if modeler is not None:
            self.modeler = modeler.lower()
            self.load_data()
        else:
            self.modeler=None
        return

    def load_data(self):
        self.datfile = None
        for extension in ['.dat','.fits']:
            datfile = os.path.join(
                thispath, 'data/' + self.modeler + extension)
            if os.path.isfile(datfile):
                self.datfile = datfile
        if self.datfile is None:
            print("Cannot find data for %s" % self.modeler)
            return

        if self.datfile.endswith('fits'):
            dat = Table.read(self.datfile)
        else:
            dat = ascii.read(self.datfile, format='commented_header',
                             data_start=0, header_start=-1)

        if self.modeler == 'oguri':
            # compute new columns from the data for plotting
            dat['mu1'] = np.abs(dat['mu1'])  # absolute value of magnifications
            dat['dt13'] = - dat['dt31']  # time delays relative to the NW event
            dat['dt12'] = dat['dt31'] - dat['dt32']
            self.modelrange = [(-10,90),(-10,250),(2,4),(-4,18),0.999]
            self.weights = None
            self.nbins = 20

        elif self.modeler == 'zitrin':
            self.modelrange = [(20,220),(10,70),(2.8,4.5),(15,70), 0.999]
            self.weights = None
            self.nbins = 15

        elif self.modeler.lower().startswith('jauzac'):
            dat['MAGS1'].name = 'mu1'
            dat['MAGS2'].name = 'mu2'
            dat['MAGS3'].name = 'mu3'
            dat['EMAGS1'].name = 'errmu1'
            dat['EMAGS2'].name = 'errmu2'
            dat['DTD_S2'].name = 'dt12'
            dat['DTD_S3'].name = 'dt13'
            self.weights = 1 / np.sqrt(dat['errmu1']**2 + dat['errmu2']**2)
            self.modelrange = [0.999, 0.999, 0.999, 0.999, 0.999]
            self.nbins = 15

        elif self.modeler == 'williams':
            dat['dt12'] = -dat['dt21']
            dat['dt13'] = -dat['dt31']
            dat['mu1'] = np.abs(dat['mu1'])
            self.weights = 1/ np.sqrt(dat['delx1']**2 + dat['dely1']**2 +
                                 dat['delx2']**2 + dat['dely2']**2)
            self.modelrange = [(-2,25), (-2,35), (0,5), (-50,10), (-4500, -300)]
            self.nbins = 12

        # reformat as an ordered array of samples for corner plotting
        newtable = Table([dat['mu1'], dat['mu2'], dat['mu3'],
                          dat['dt12'], dat['dt13']])
        newdat = np.array([list(d) for d in newtable.as_array()])
        self.dat = newdat
        self.table = newtable
        return

    def mk_corner_plot(self):
        labels = ['$\mu_{\\rm NW}$', '$\mu_{\\rm SE}$', '$\mu_{\\rm 11.3}$',
                  '$\Delta t_{\\rm NW:SE}$','$\Delta t_{\\rm NW:11.3}$',
                  ]

        corner.corner(self.dat, bins=self.nbins, range=self.modelrange,
                      weights=self.weights, color=u'k',
                      smooth=None, smooth1d=None,
                      labels=labels, label_kwargs={'fontsize':'large'},
                      show_titles=False, title_fmt=u'.2f', title_kwargs=None,
                      truths=None, truth_color=u'#4682b4',
                      scale_hist=False, quantiles=[0.16,0.84],
                      verbose=False, fig=None, max_n_ticks=5,
                      top_ticks=False, use_math_text=False, hist_kwargs=None)

        # print out the mean and standard deviation from the corner plots
        modelresults = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                         zip(*np.percentile(self.dat, [16, 50, 84], axis=0)))
        for label, modres in zip(labels, modelresults):
            print '%s = %.2f +%.2f -%.2f' % (
                label, modres[0], modres[1], modres[2])



def mk_composite_model():
    modeldict = {}
    for modeler in ['oguri','zitrin','jauzac','williams']:
        modeldict[modeler] = LensModel(modeler=modeler)
    return(modeldict)

