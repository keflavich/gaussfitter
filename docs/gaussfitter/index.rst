***********
Gaussfitter
***********

Documentation for gaussfitter.

Simple Example
--------------

.. code-block:: python

   from astropy.io import fits
   from astropy import wcs
   from gaussfitter import gaussfit

   file = fits.open('file.fits')

   # generate the world coordinate system
   mywcs = wcs.WCS(file[0].header)
   data = file[0].data

   # cut out a subset containing a gaussian & fit it
   x0, xrange = 50, 40
   y0, yrange = 60, 40
   height,amp,xfit,yfit,xwid,ywid,angle = gaussfitter(data[y0:y0+yrange, x0:x0+xrange])

   # convert back to the original file's pixel units
   xc, yc = xfit+x0, yfit+y0

   # convert to world coordinates
   ra, dec = mywcs.all_pix2world(xc, yc, 0)

Another Simple Example
----------------------

.. code-block:: python

    from gaussfitter import gaussfit
    from astropy.io import fits
    import pylab as pl
    d = fits.getdata('W51e2_99_pa90.pv.fits')
    pars, im = gaussfit(d, returnfitimage=True)

    from astropy import wcs
    mywcs = wcs.WCS(fits.getheader('W51e2_99_pa90.pv.fits'))
    xc,yc = mywcs.all_pix2world(pars[2],pars[3],0)

    pl.clf()
    pl.imshow(d, cmap=pl.cm.gray)
    pl.contour(im, cmap=pl.cm.Spectral)
    pl.annotate("x0 = {0:0.2f} pix".format(pars[2]), xy=(0.75, 0.7), xycoords='figure fraction',)
    pl.annotate("y0 = {0:0.2f} km/s".format(float(yc)/1e3), xy=(0.75, 0.56), xycoords='figure fraction',)
    pl.annotate("y0 = {0:0.2f} pix".format(pars[3]), xy=(0.75, 0.6), xycoords='figure fraction',)
    pl.annotate("x0 = {0:0.2f} arcsec".format(float(xc)), xy=(0.75, 0.66), xycoords='figure fraction',)
    pl.savefig("pv_gfit.png")

An example comparing with astropy's modeling
--------------------------------------------

.. code-block:: python

    from astropy.io import fits
    import numpy as np
    from astropy import wcs

    im = fits.getdata('IC860CbandCarrayH2COJ1Cont.fits')

    from gaussfitter import gaussfit
    from astropy.modeling import models,fitting

    M = models.Gaussian2D(0.02, 16, 14, 3, 3)
    lmf = fitting.LevMarLSQFitter()

    result_gf, fitim_gf = gaussfit(im[450:480, 450:480], returnfitimage=True)

    xx,yy = np.indices([30,30])
    result_lmf = lmf(M, xx, yy, im[450:480, 450:480])

    w = wcs.WCS(fits.getheader('IC860CbandCarrayH2COJ1Cont.fits'))

    w.all_pix2world(result_gf[2]+450, result_gf[3]+450, 0)
    w.all_pix2world(result_lmf.parameters[1]+450, result_lmf.parameters[2]+450, 0)

    fitim_lmf = result_lmf(xx,yy)

    import pylab as pl
    pl.imshow(fitim_gf, cmap=pl.cm.gray)
    pl.contour(fitim_lmf)

Reference/API
=============

.. automodapi:: gaussfitter
    :no-heading:
