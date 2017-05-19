from astropy.io import ascii
from astropy.table import Table
import numpy as np
import os
import sys
import corner
from matplotlib import pyplot as pl
from matplotlib import rcParams
from astropy import table


thisfile = sys.argv[0]
if 'ipython' in thisfile : thisfile = __file__
thispath = os.path.abspath( os.path.dirname( thisfile ) )

_MONTECARLODIR_ = "data/MonteCarloChains/"
_MODEL_LIST_ = ['CATS-A', 'ZLTM','GLAFIC-A','GLEE-A', 'GRALE' ]
_USETEX_ = rcParams['text.usetex']

class LensModel(object):
    def __init__(self, modelname=None):
        self.modelrange = [0.999, 0.999, 0.999, 0.999, 0.999]
        self.weights = None
        self.nbins = 20
        if modelname is not None:
            self.modelname = modelname.lower()
            self.load_data()
        else:
            self.modelname=None
        return

    def load_data(self):
        self.datfile = None
        for extension in ['_chain.dat','_chain.txt','_chain.fits']:
            datfile = os.path.join(
                thispath, _MONTECARLODIR_ + self.modelname + extension)
            if os.path.isfile(datfile):
                self.datfile = datfile
        if self.datfile is None:
            print("Cannot find data for %s" % self.modelname)
            return

        if self.datfile.endswith('fits'):
            dat = Table.read(self.datfile)
        else:
            dat = ascii.read(self.datfile, format='commented_header',
                             data_start=0, header_start=-1)

        if self.modelname.upper().startswith('GLAFIC'):
            # compute new columns from the data for plotting
            dat['mu1'] = np.abs(dat['mu1'])  # absolute value of magnifications
            dat['dt13'] = - dat['dt31']  # time delays relative to the NW event
            dat['dt12'] = dat['dt31'] - dat['dt32']
            self.modelrange = [(-10,90),(-10,250),(2,4),(-4,18),0.999]
            self.weights = None
            self.nbins = 20

        elif self.modelname.upper().startswith('ZLTM'):
            self.modelrange = [(20,220),(10,70),(2.8,4.5),(15,70), 0.999]
            self.weights = None
            self.nbins = 15

        elif self.modelname.upper().startswith('CATS'):
            dat['MAGS1'].name = 'mu1'
            dat['MAGS2'].name = 'mu2'
            dat['MAGS3'].name = 'mu3'
            #dat['EMAGS1'].name = 'errmu1'
            #dat['EMAGS2'].name = 'errmu2'
            dat['DTD_S2'].name = 'dt12'
            dat['DTD_S3'].name = 'dt13'
            # self.weights = 1 / np.sqrt(dat['errmu1']**2 + dat['errmu2']**2)
            self.modelrange = [(90,350), (42,55), (3.1,3.6), (-6,3), (-4.55, -3.)]
            self.modelrange = [0.95, 0.95, 0.95, 0.95, 0.95]

            self.nbins = 15

        elif self.modelname.upper().startswith('GRALE'):
            dat['dt12'] = -dat['dt21']
            dat['dt13'] = -dat['dt31']
            dat['mu1'] = np.abs(dat['mu1'])
            #self.weights = 1/ np.sqrt(dat['delx1']**2 + dat['dely1']**2 +
            #                     dat['delx2']**2 + dat['dely2']**2)
            self.modelrange = [(-2,25), (-2,35), (0,5), (-50,10), (-12, -0.5)]
            self.nbins = 12

        elif self.modelname.upper().startswith('GLEE'):
            dat['t_2o'].name = 'dt12'
            dat['t_3o'].name = 'dt13'
            dat['mu1'] = np.abs(dat['m_1o'])
            dat['mu2'] = np.abs(dat['m_2o'])
            dat['mu3'] = np.abs(dat['m_3o'])
            #self.modelrange = [(-2,25), (-2,35), (0,5), (-50,10), (-12, -0.5)]
            self.modelrange = [(10,350), (35,150), 0.95, 0.95, 0.95]
            self.nbins = 10

        # reformat as an ordered array of samples for corner plotting
        newtable = Table([dat['mu1'], dat['mu2'], dat['mu3'],
                          dat['dt12'], dat['dt13']/365.])
        newdat = np.array([list(d) for d in newtable.as_array()])
        self.data = newdat
        self.table = newtable
        return

    def mk_corner_plot(self, **kwarg):
        if _USETEX_:
            labels = ['$\mu_{\\rm NW}$', '$\mu_{\\rm SE}$',
                      '$\mu_{\\rm 11.3}$',
                      '$\Delta t_{\\rm NW:SE}$~[days]',
                      '$\Delta t_{\\rm NW:11.3}$~[yrs]',
                      ]
        else:
            labels = ['muNW', 'muSE', 'mu11.3',
                      'dt NW:SE [days]', 'dt NW:11.3 [yrs]',
                      ]

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
                mnstrtuple = (label, round(modres[0], 1),
                                   round(modres[1], 1), round(modres[2], 1),
                                   units)
                if _USETEX_:
                    meanstring = '%s = %.1f $^{+%.1f}_{-%.1f}$ %s' % mnstrtuple
                else:
                    meanstring = '%s = %.1f +%.1f -%.1f %s' % mnstrtuple
            else:
                mnstrtuple = (label, round(modres[0]), round(modres[1]),
                              round(modres[2]), units)
                if _USETEX_:
                    meanstring = '%s = %i $^{+%i}_{-%i}$ %s' % mnstrtuple
                else:
                    meanstring = '%s = %i +%i -%i %s' % mnstrtuple
            print(meanstring)
            fig.text(0.6, ytext, meanstring, fontsize='x-large',
                     ha='left', va='top', transform=fig.transFigure)
            ytext -= 0.05

        # also report the observed time between events
        dtobsstring = '$\Delta t_{\\rm NW:SE, observed}$~=~$234\pm6$~days'
        fig.text(0.6, ytext, dtobsstring, fontsize='x-large',
                 ha='left', va='top', transform=fig.transFigure)


def mk_composite_model():
    combomodel = LensModel(modelname=None)

    # define ranges for the 5 model parameters:
    #  muNW, muSE, mu11.3, dt_NW:SE, dt_NW:11.3,
    combomodel.modelrange = [(-10,350),(-10,120),(2.5,4.5),(-25,75),(-7,-3)]

    # and set the default number of bins to use in corner plots
    combomodel.nbins = 25

    # read in and join the data from all models
    combotable = None
    weights = np.array([])
    for modeler in _MODEL_LIST_:
        lensmodel = LensModel(modelname=modeler)
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

