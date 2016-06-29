from astropy import cosmology
import os
import sys

__MJDPKNW__ = 56672.7
__MJDPKSE__ = 56901.5
__Z__ = 1.0054

__MJDPREPK0NW__ = __MJDPKNW__ - 6 * (1 + __Z__)
__MJDPOSTPK0NW__ = __MJDPKNW__ + 2 * (1 + __Z__)

__MJDPREPK0SE__ = __MJDPKSE__ - 6 * (1 + __Z__)
__MJDPOSTPK0SE__ = __MJDPKSE__ + 7.2 * (1 + __Z__)

__H0__ = 70
__OM__ = 0.3

cosmo = cosmology.FlatLambdaCDM( name="WMAP9", H0=__H0__, Om0=__OM__ )
__DM__ = cosmo.distmod( __Z__ ).value

__ABRESTBANDNAME__ = {'f435w':'sdssu', 'f606w':'sdssu',
                      'f814w':'sdssg', 'f105w':'sdssg',
                      'f125w':'sdssr', 'f140w':'sdssr',
                      'f160w':'sdssi'}

__VEGARESTBANDNAME__ = {'f435w':'bessellux', 'f606w':'bessellux',
                        'f814w':'bessellb', 'f105w':'bessellv',
                        'f125w':'bessellr', 'f140w':'bessellr',
                        'f160w':'besselli'}

__THISFILE__ = sys.argv[0]
if 'ipython' in __THISFILE__:
    __THISFILE__ = __file__
__THISDIR__ = os.path.abspath(os.path.dirname(__THISFILE__))

