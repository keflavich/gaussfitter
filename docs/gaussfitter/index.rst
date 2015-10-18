***********
Gaussfitter
***********

Documentation for gaussfitter.

Simple Example
--------------

.. code-block::

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
   height,amp,xfit,yfit,xwid,ywid,angle) = gaussfitter(data[y0:y0+yrange, x0:x0+xrange])

   # convert back to the original file's pixel units
   xc, yc = xfit+x0, yfit+y0

   # convert to world coordinates
   ra, dec = mywcs.all_pix2world(xc, yc, 0)


Reference/API
=============

.. automodapi:: packagename
