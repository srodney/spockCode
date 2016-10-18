from astropy.io import ascii
from astropy.table import Table
import numpy as np
import os
import sys
import corner
from matplotlib import pyplot as pl
from astropy import table

thisfile = sys.argv[0]
if 'ipython' in thisfile : thisfile = __file__
thispath = os.path.abspath( os.path.dirname( thisfile ) )


class LensModel(object):
    def __init__(self, modeler=None):
        self.modelrange = [0.999, 0.999, 0.999, 0.999, 0.999]
        self.weights = None
        self.nbins = 20
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
            # self.weights = 1 / np.sqrt(dat['errmu1']**2 + dat['errmu2']**2)
            self.modelrange = [0.999, 0.999, 0.999, 0.999, 0.999]
            self.nbins = 15

        elif self.modeler == 'williams':
            dat['dt12'] = -dat['dt21']
            dat['dt13'] = -dat['dt31']
            dat['mu1'] = np.abs(dat['mu1'])
            #self.weights = 1/ np.sqrt(dat['delx1']**2 + dat['dely1']**2 +
            #                     dat['delx2']**2 + dat['dely2']**2)
            self.modelrange = [(-2,25), (-2,35), (0,5), (-50,10), (-12, -0.5)]
            self.nbins = 12

        # reformat as an ordered array of samples for corner plotting
        newtable = Table([dat['mu1'], dat['mu2'], dat['mu3'],
                          dat['dt12'], dat['dt13']/365.])
        newdat = np.array([list(d) for d in newtable.as_array()])
        self.data = newdat
        self.table = newtable
        return

    def mk_corner_plot(self, **kwarg):
        labels = ['$\mu_{\\rm NW}$', '$\mu_{\\rm SE}$', '$\mu_{\\rm 11.3}$',
                  '$\Delta t_{\\rm NW:SE}$~[days]','$\Delta t_{\\rm NW:11.3}$~[yrs]',
                  ]
        #labels = ['muNW', 'muSE', 'mu11.3',
        #          'dt NW:SE [days]', 'dt NW:11.3 [yrs]',
        #          ]

        levels = 1.0 - np.exp(-0.5 * np.array([1.0,2.0]) ** 2)
        corner.corner(self.data, bins=self.nbins, range=self.modelrange,
                      levels=levels,
                      weights=self.weights, quantiles=[0.16, 0.5, 0.84],
                      labels=labels, label_kwargs={'fontsize':'large'},
                      # show_titles=True, title_kwargs={'fontsize':'x-large'},
                      # title_fmt='.1f',
                      plot_contours=True, plot_datapoints=False,
                      fill_contours=True,
                      **kwarg)

        # print out the mean and standard deviation from the corner plots
        fig = pl.gcf()
        modelresults = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                           zip(*np.percentile(self.data, [16, 50, 84], axis=0)))
        ytext = 0.9
        for label, modres, units  in zip(labels, modelresults,
                                         ['','','','days','years']):
            #if modres == modelresults[-1]:
            #    modres = [m/365. for m in modres]
            if abs(modres[0])<10:
                meanstring = '%s = %.1f $^{+%.1f}_{-%.1f}$ %s' % (
                    label, round(modres[0],1), round(modres[1],1),
                    round(modres[2],1), units)
            else:
                meanstring = '%s = %i $^{+%i}_{-%i}$ %s' % (
                    label, round(modres[0]), round(modres[1]),
                    round(modres[2]), units)
            print(meanstring)
            fig.text(0.6, ytext, meanstring, fontsize='x-large',
                     ha='left', va='top', transform=fig.transFigure)
            ytext -= 0.05

def mk_composite_model():
    combomodel = LensModel(modeler=None)

    # define ranges for the 5 model parameters:
    #  muNW, muSE, mu11.3, dt_NW:SE, dt_NW:11.3,
    combomodel.modelrange = [(-10,150),(-10,150),(2,5),(-30,75),(-7,0)]

    # and set the default number of bins to use in corner plots
    combomodel.nbins = 25

    # read in and join the data from all models
    combotable = None
    weights = np.array([])
    for modeler in ['oguri', 'zitrin','jauzacB','williams']:
        lensmodel = LensModel(modeler=modeler)
        if combotable is None:
            combotable = lensmodel.table
        else:
            combotable = table.join(combotable, lensmodel.table,
                                    join_type='outer')
        weights = np.append(
            weights, np.zeros(len(lensmodel.data)) + 1.0 / len(lensmodel.data))

    combomodel.table = combotable
    combomodel.data = np.array([list(d) for d in combotable.as_array()])
    combomodel.weights = None # weights

    return(combomodel)

