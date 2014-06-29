"""
===========
gaussfitter
===========
.. codeauthor:: Adam Ginsburg <adam.g.ginsburg@gmail.com> 3/17/08

Latest version available at <http://code.google.com/p/agpy/source/browse/trunk/agpy/gaussfitter.py>

As of January 30, 2014, gaussfitter has its own code repo on github:
    https://github.com/keflavich/gaussfitter

"""
import numpy as np
from numpy.ma import median
from numpy import pi
from .mpfit import mpfit
"""
Note about mpfit/leastsq:
I switched everything over to the Markwardt mpfit routine for a few reasons,
but foremost being the ability to set limits on parameters, not just force them
to be fixed.  As far as I can tell, leastsq does not have that capability.

The version of mpfit I use can be found here:
    http://code.google.com/p/agpy/source/browse/trunk/mpfit

Alternative: lmfit

.. todo::
    -turn into a class instead of a collection of objects
    -implement WCS-based gaussian fitting with correct coordinates
"""


def moments(data, circle, rotate, vheight, estimator=median, **kwargs):
    """Returns (height, amplitude, x, y, width_x, width_y, rotation angle)
    the gaussian parameters of a 2D distribution by calculating its
    moments.  Depending on the input parameters, will only output
    a subset of the above.

    If using masked arrays, pass estimator=np.ma.median
    """
    total = np.abs(data).sum()
    Y, X = np.indices(data.shape)  # python convention: reverse x,y np.indices
    y = np.argmax((X*np.abs(data)).sum(axis=1)/total)
    x = np.argmax((Y*np.abs(data)).sum(axis=0)/total)
    col = data[int(y), :]
    # FIRST moment, not second!
    width_x = np.sqrt(np.abs((np.arange(col.size)-y)*col).sum() / np.abs(col).sum())
    row = data[:, int(x)]
    width_y = np.sqrt(np.abs((np.arange(row.size)-x)*row).sum() / np.abs(row).sum())
    width = (width_x + width_y) / 2.
    height = estimator(data.ravel())
    amplitude = data.max()-height
    mylist = [amplitude, x, y]
    if (np.isnan(width_y) or np.isnan(width_x) or np.isnan(height) or np.isnan(amplitude)):
        raise ValueError("something is nan")
    if vheight:
        mylist = [height] + mylist
    if not circle:
        mylist = mylist + [width_x, width_y]
        if rotate:
            mylist = mylist + [0.]  # rotation "moment" is just zero...
            # also, circles don't rotate.
    else:
        mylist = mylist + [width]
    return mylist


def twodgaussian(inpars, circle=False, rotate=True, vheight=True, shape=None):
    """Returns a 2d gaussian function of the form:
        x' = np.cos(rota) * x - np.sin(rota) * y
        y' = np.sin(rota) * x + np.cos(rota) * y
        (rota should be in degrees)
        g = b + a * np.exp ( - ( ((x-center_x)/width_x)**2 +
        ((y-center_y)/width_y)**2 ) / 2 )

        inpars = [b,a,center_x,center_y,width_x,width_y,rota]
                 (b is background height, a is peak amplitude)

        where x and y are the input parameters of the returned function,
        and all other parameters are specified by this function

        However, the above values are passed by list.  The list should be:
        inpars = (height,amplitude,center_x,center_y,width_x,width_y,rota)

        You can choose to ignore / neglect some of the above input parameters
            unp.sing the following options:
            circle=0 - default is an elliptical gaussian (different x, y
                widths), but can reduce the input by one parameter if it's a
                circular gaussian
            rotate=1 - default allows rotation of the gaussian ellipse.  Can
                remove last parameter by setting rotate=0
            vheight=1 - default allows a variable height-above-zero, i.e. an
                additive constant for the Gaussian function.  Can remove first
                parameter by setting this to 0
            shape=None - if shape is set (to a 2-parameter list) then returns
                an image with the gaussian defined by inpars
        """
    inpars_old = inpars
    inpars = list(inpars)
    if vheight:
        height = inpars.pop(0)
        height = float(height)
    else:
        height = float(0)
    amplitude, center_y, center_x = inpars.pop(0), inpars.pop(0), inpars.pop(0)
    amplitude = float(amplitude)
    center_x = float(center_x)
    center_y = float(center_y)
    if circle:
        width = inpars.pop(0)
        width_x = float(width)
        width_y = float(width)
        rotate = 0
    else:
        width_x, width_y = inpars.pop(0), inpars.pop(0)
        width_x = float(width_x)
        width_y = float(width_y)
    if rotate:
        rota = inpars.pop(0)
        rota = pi/180. * float(rota)
        rcen_x = center_x * np.cos(rota) - center_y * np.sin(rota)
        rcen_y = center_x * np.sin(rota) + center_y * np.cos(rota)
    else:
        rcen_x = center_x
        rcen_y = center_y
    if len(inpars) > 0:
        raise ValueError("There are still input parameters:" + str(inpars) +
                         " and you've input: " + str(inpars_old) +
                         " circle=%d, rotate=%d, vheight=%d" % (circle, rotate, vheight))

    def rotgauss(x, y):
        if rotate:
            xp = x * np.cos(rota) - y * np.sin(rota)
            yp = x * np.sin(rota) + y * np.cos(rota)
        else:
            xp = x
            yp = y
        g = height+amplitude*np.exp(-(((rcen_x-xp)/width_x)**2 +
                                      ((rcen_y-yp)/width_y)**2)/2.)
        return g
    if shape is not None:
        return rotgauss(*np.indices(shape))
    else:
        return rotgauss


def gaussfit(data, err=None, params=(), autoderiv=True, return_all=False,
             circle=False, fixed=np.repeat(False, 7),
             limitedmin=[False, False, False, False, True, True, True],
             limitedmax=[False, False, False, False, False, False, True],
             usemoment=np.array([], dtype='bool'),
             minpars=np.repeat(0, 7), maxpars=[0, 0, 0, 0, 0, 0, 360],
             rotate=True, vheight=True, quiet=True, returnmp=False,
             returnfitimage=False, **kwargs):
    """
    Gaussian fitter with the ability to fit a variety of different forms of
    2-dimensional gaussian.

    Input Parameters:
        data - 2-dimensional data array
        err=None - error array with same size as data array
        params=[] - initial input parameters for Gaussian function.
            (height, amplitude, x, y, width_x, width_y, rota)
            if not input, these will be determined from the moments of the
            system, assuming no rotation
        autoderiv=1 - use the autoderiv provided in the lmder.f function (the
            alternative is to us an analytic derivative with lmdif.f: this
            method is less robust)
        return_all=0 - Default is to return only the Gaussian parameters.
                   1 - fit params, fit error
        returnfitimage - returns (best fit params,best fit image)
        returnmp - returns the full mpfit struct
        circle=0 - default is an elliptical gaussian (different x, y widths),
            but can reduce the input by one parameter if it's a circular gaussian
        rotate=1 - default allows rotation of the gaussian ellipse.  Can remove
            last parameter by setting rotate=0.  np.expects angle in DEGREES
        vheight=1 - default allows a variable height-above-zero, i.e. an
            additive constant for the Gaussian function.  Can remove first
            parameter by setting this to 0
        usemoment - can choose which parameters to use a moment estimation for.
            Other parameters will be taken from params.  Needs to be a boolean
            array.

    Output:
        Default output is a set of Gaussian parameters with the same shape as
            the input parameters

        If returnfitimage=True returns a np array of a gaussian
            contructed using the best fit parameters.

        If returnmp=True returns a `mpfit` object. This object contains
            a `covar` attribute which is the 7x7 covariance array
            generated by the mpfit class in the `mpfit_custom.py`
            module. It contains a `param` attribute that contains a
            list of the best fit parameters in the same order as the
            optional input parameter `params`.

        The rotation angle will always be 0 <= theta < 360 degrees.
    """
    data = data.view(np.ma.MaskedArray)
    usemoment = np.array(usemoment, dtype='bool')
    params = np.array(params, dtype='float')
    if usemoment.any() and len(params) == len(usemoment):
        moment = np.array(moments(data, circle, rotate, vheight, **kwargs), dtype='float')
        params[usemoment] = moment[usemoment]
    elif params == [] or len(params) == 0:
        params = (moments(data, circle, rotate, vheight, **kwargs))
    if not vheight:
        vheight = True
        params = np.concatenate([[0], params])
        fixed[0] = 1

    # mpfit will fail if it is given a start parameter outside the allowed range:
    for i in xrange(len(params)):
        if params[i] > maxpars[i] and limitedmax[i]: params[i] = maxpars[i]
        if params[i] < minpars[i] and limitedmin[i]: params[i] = minpars[i]

    def mpfitfun(data, err):
        err = err if err is not None else 1.0
        def f(p, fjac):
            twodg = twodgaussian(p, circle, rotate, vheight)
            delta = (data - twodg(*np.indices(data.shape))) / err
            return [0, delta.compressed()]
        return f

    parinfo = [{'n': 1, 'value': params[1], 'limits': [minpars[1], maxpars[1]],
                'limited': [limitedmin[1], limitedmax[1]], 'fixed': fixed[1],
                'parname': "AMPLITUDE", 'error': 0},
               {'n': 2, 'value': params[2], 'limits': [minpars[2], maxpars[2]],
                'limited': [limitedmin[2], limitedmax[2]], 'fixed': fixed[2],
                'parname': "XSHIFT", 'error': 0},
               {'n': 3, 'value': params[3], 'limits': [minpars[3], maxpars[3]],
                'limited': [limitedmin[3], limitedmax[3]], 'fixed': fixed[3],
                'parname': "YSHIFT", 'error': 0},
               {'n': 4, 'value': params[4], 'limits': [minpars[4], maxpars[4]],
                'limited': [limitedmin[4], limitedmax[4]], 'fixed': fixed[4],
                'parname': "XWIDTH", 'error': 0}]
    if vheight:
        parinfo.insert(0, {'n': 0, 'value': params[0], 'limits': [minpars[0], maxpars[0]],
                           'limited': [limitedmin[0], limitedmax[0]], 'fixed': fixed[0],
                           'parname': "HEIGHT", 'error': 0})
    if not circle:
        parinfo.append({'n': 5, 'value': params[5], 'limits': [minpars[5], maxpars[5]],
                        'limited': [limitedmin[5], limitedmax[5]], 'fixed': fixed[5],
                        'parname': "YWIDTH", 'error': 0})
        if rotate:
            parinfo.append({'n': 6, 'value': params[6], 'limits': [minpars[6], maxpars[6]],
                            'limited': [limitedmin[6], limitedmax[6]], 'fixed': fixed[6],
                            'parname': "ROTATION", 'error': 0})

    if not autoderiv:
        # the analytic derivative, while not terribly difficult, is less
        # efficient and useful.  I only bothered putting it here because I was
        # instructed to do so for a class project - please ask if you would
        # like this feature implemented
        raise ValueError("I'm sorry, I haven't implemented this feature yet.")
    else:
        mp = mpfit(mpfitfun(data, err), parinfo=parinfo, quiet=quiet)

    if (not circle) and rotate:
        mp.params[-1] %= 360.0

    if returnmp:
        returns = (mp)
    elif return_all == 0:
        returns = mp.params
    elif return_all == 1:
        returns = mp.params, mp.perror
    if returnfitimage:
        fitimage = twodgaussian(mp.params, circle, rotate, vheight)(*np.indices(data.shape))
        returns = (returns, fitimage)
    return returns


def onedmoments(Xax, data, vheight=True, estimator=median, negamp=None, veryverbose=False,
                **kwargs):
    """Returns (height, amplitude, x, width_x)
    the gaussian parameters of a 1D distribution by calculating its
    moments.  Depending on the input parameters, will only output
    a subset of the above.

    If using masked arrays, pass estimator=np.ma.median
    'estimator' is used to measure the background level (height)

    negamp can be used to force the peak negative (True), positive (False),
    or it will be "autodetected" (negamp=None)
    """

    dx = np.mean(Xax[1:] - Xax[:-1])  # assume a regular grid
    integral = (data*dx).sum()
    height = estimator(data)

    # try to figure out whether pos or neg based on the minimum width of the pos/neg peaks
    Lpeakintegral = integral - height*len(Xax)*dx - (data[data > height]*dx).sum()
    Lamplitude = data.min()-height
    Lwidth_x = 0.5*(np.abs(Lpeakintegral / Lamplitude))
    Hpeakintegral = integral - height*len(Xax)*dx - (data[data < height]*dx).sum()
    Hamplitude = data.max()-height
    Hwidth_x = 0.5*(np.abs(Hpeakintegral / Hamplitude))
    Lstddev = Xax[data < data.mean()].std()
    Hstddev = Xax[data > data.mean()].std()
    # print "Lstddev: %10.3g  Hstddev: %10.3g" % (Lstddev,Hstddev)
    # print "Lwidth_x: %10.3g  Hwidth_x: %10.3g" % (Lwidth_x,Hwidth_x)

    if negamp:  # can force the guess to be negative
        xcen, amplitude, width_x = Xax[np.argmin(data)], Lamplitude, Lwidth_x
    elif negamp is None:
        if Hstddev < Lstddev:
            xcen, amplitude, width_x, = Xax[np.argmax(data)], Hamplitude, Hwidth_x
        else:
            xcen, amplitude, width_x, = Xax[np.argmin(data)], Lamplitude, Lwidth_x
    else:  # if negamp==False, make positive
        xcen, amplitude, width_x = Xax[np.argmax(data)], Hamplitude, Hwidth_x

    if veryverbose:
        print "negamp: %s  amp,width,cen Lower: %g, %g   Upper: %g, %g  Center: %g" %\
              (negamp, Lamplitude, Lwidth_x, Hamplitude, Hwidth_x, xcen)
    mylist = [amplitude, xcen, width_x]
    if np.isnan(width_x) or np.isnan(height) or np.isnan(amplitude):
        raise ValueError("something is nan")
    if vheight:
        mylist = [height] + mylist
    return mylist


def onedgaussian(x, H, A, dx, w):
    """
    Returns a 1-dimensional gaussian of form
    H+A*np.exp(-(x-dx)**2/(2*w**2))
    """
    return H+A*np.exp(-(x-dx)**2/(2*w**2))


def onedgaussfit(xax, data, err=None,
                 params=[0, 1, 0, 1], fixed=[False, False, False, False],
                 limitedmin=[False, False, False, True], limitedmax=[False, False, False, False],
                 minpars=[0, 0, 0, 0], maxpars=[0, 0, 0, 0],
                 quiet=True, shh=True, veryverbose=False,
                 vheight=True, negamp=False, usemoments=False):
    """
    Inputs:
       xax - x axis
       data - y axis
       err - error corresponding to data

       params - Fit parameters: Height of background, Amplitude, Shift, Width
       fixed - Is parameter fixed?
       limitedmin/minpars - set lower limits on each parameter (default: width>0)
       limitedmax/maxpars - set upper limits on each parameter
       quiet - should MPFIT output each iteration?
       shh - output final parameters?
       usemoments - replace default parameters with moments

    Returns:
       Fit parameters
       Model
       Fit errors
       chi2
    """

    def mpfitfun(x, y, err):
        if err is None:
            def f(p, fjac=None): return [0, (y-onedgaussian(x, *p))]
        else:
            def f(p, fjac=None): return [0, (y-onedgaussian(x, *p))/err]
        return f

    if xax is None:
        xax = np.arange(len(data))

    if not vheight:
        height = params[0]
        fixed[0] = True
    if usemoments:
        params = onedmoments(xax, data, vheight=vheight, negamp=negamp, veryverbose=veryverbose)
        if vheight is False:
            params = [height]+params
        if veryverbose:
            print "OneD moments: h: %g  a: %g  c: %g  w: %g" % tuple(params)

    parinfo = [{'n': 0, 'value': params[0], 'limits': [minpars[0], maxpars[0]],
                'limited': [limitedmin[0], limitedmax[0]], 'fixed': fixed[0],
                'parname': "HEIGHT", 'error': 0},
               {'n': 1, 'value': params[1], 'limits': [minpars[1], maxpars[1]],
                'limited': [limitedmin[1], limitedmax[1]], 'fixed': fixed[1],
                'parname': "AMPLITUDE", 'error': 0},
               {'n': 2, 'value': params[2], 'limits': [minpars[2], maxpars[2]],
                'limited': [limitedmin[2], limitedmax[2]], 'fixed': fixed[2],
                'parname': "SHIFT", 'error': 0},
               {'n': 3, 'value': params[3], 'limits': [minpars[3], maxpars[3]],
                'limited': [limitedmin[3], limitedmax[3]], 'fixed': fixed[3],
                'parname': "WIDTH", 'error': 0}]

    mp = mpfit(mpfitfun(xax, data, err), parinfo=parinfo, quiet=quiet)
    mpp = mp.params
    mpperr = mp.perror
    chi2 = mp.fnorm

    if mp.status == 0:
        raise Exception(mp.errmsg)

    if (not shh) or veryverbose:
        print "Fit status: ", mp.status
        for i, p in enumerate(mpp):
            parinfo[i]['value'] = p
            print parinfo[i]['parname'], p, " +/- ", mpperr[i]
        print "Chi2: ", mp.fnorm, " Reduced Chi2: ", mp.fnorm/len(data), " DOF:", len(data)-len(mpp)

    return mpp, onedgaussian(xax, *mpp), mpperr, chi2


def n_gaussian(pars=None, a=None, dx=None, sigma=None):
    """
    Returns a function that sums over N gaussians, where N is the length of
    a,dx,sigma *OR* N = len(pars) / 3

    The background "height" is assumed to be zero (you must "baseline" your
    spectrum before fitting)

    pars  - a list with len(pars) = 3n, assuming a,dx,sigma repeated
    dx    - offset (velocity center) values
    sigma - line widths
    a     - amplitudes
    """
    if len(pars) % 3 == 0:
        a = [pars[ii] for ii in xrange(0, len(pars), 3)]
        dx = [pars[ii] for ii in xrange(1, len(pars), 3)]
        sigma = [pars[ii] for ii in xrange(2, len(pars), 3)]
    elif not(len(dx) == len(sigma) == len(a)):
        raise ValueError("Wrong array lengths! dx: %i  sigma: %i  a: %i" %
                         (len(dx), len(sigma), len(a)))

    def g(x):
        v = np.zeros(len(x))
        for i in range(len(dx)):
            v += a[i] * np.exp(-(x-dx[i])**2 / (2.0*sigma[i]**2))
        return v
    return g


def multigaussfit(xax, data, ngauss=1, err=None, params=[1, 0, 1],
                  fixed=[False, False, False], limitedmin=[False, False, True],
                  limitedmax=[False, False, False], minpars=[0, 0, 0], maxpars=[0, 0, 0],
                  quiet=True, shh=True, veryverbose=False):
    """
    An improvement on onedgaussfit.  Lets you fit multiple gaussians.

    Inputs:
       xax - x axis
       data - y axis
       ngauss - How many gaussians to fit?  Default 1 (this could supersede onedgaussfit)
       err - error corresponding to data

     These parameters need to have length = 3*ngauss.  If ngauss > 1 and length = 3, they will
     be replicated ngauss times, otherwise they will be reset to defaults:
       params - Fit parameters: [amplitude, offset, width] * ngauss
              If len(params) % 3 == 0, ngauss will be set to len(params) / 3
       fixed - Is parameter fixed?
       limitedmin/minpars - set lower limits on each parameter (default: width>0)
       limitedmax/maxpars - set upper limits on each parameter

       quiet - should MPFIT output each iteration?
       shh - output final parameters?

    Returns:
       Fit parameters
       Model
       Fit errors
       chi2
    """

    if len(params) != ngauss and (len(params) / 3) > ngauss:
        ngauss = len(params) / 3

    if isinstance(params, np.ndarray):
        params = params.tolist()

    # make sure all various things are the right length; if they're not, fix them using the defaults
    for parlist in (params, fixed, limitedmin, limitedmax, minpars, maxpars):
        if len(parlist) != 3*ngauss:
            # if you leave the defaults, or enter something that can be multiplied by 3 to get
            # to the right number of gaussians, it will just replicate
            if len(parlist) == 3:
                parlist *= ngauss
            elif parlist == params:
                parlist[:] = [1, 0, 1] * ngauss
            elif parlist == fixed or parlist == limitedmax:
                parlist[:] = [False, False, False] * ngauss
            elif parlist == limitedmin:
                parlist[:] = [False, False, True] * ngauss
            elif parlist == minpars or parlist == maxpars:
                parlist[:] = [0, 0, 0] * ngauss

    def mpfitfun(x, y, err):
        if err is None:
            def f(p, fjac=None): return [0, (y-n_gaussian(pars=p)(x))]
        else:
            def f(p, fjac=None): return [0, (y-n_gaussian(pars=p)(x))/err]
        return f

    if xax is None:
        xax = np.arange(len(data))

    parnames = {0: "AMPLITUDE", 1: "SHIFT", 2: "WIDTH"}

    parinfo = [{'n': ii, 'value': params[ii],
                'limits': [minpars[ii], maxpars[ii]],
                'limited': [limitedmin[ii], limitedmax[ii]], 'fixed': fixed[ii],
                'parname': parnames[ii % 3]+str(ii % 3), 'error': ii}
               for ii in xrange(len(params))]

    if veryverbose:
        print "GUESSES: "
        print "\n".join(["%s: %s" % (p['parname'], p['value']) for p in parinfo])

    mp = mpfit(mpfitfun(xax, data, err), parinfo=parinfo, quiet=quiet)
    mpp = mp.params
    mpperr = mp.perror
    chi2 = mp.fnorm

    if mp.status == 0:
        raise Exception(mp.errmsg)

    if not shh:
        print "Final fit values: "
        for i, p in enumerate(mpp):
            parinfo[i]['value'] = p
            print parinfo[i]['parname'], p, " +/- ", mpperr[i]
        print "Chi2: ", mp.fnorm, " Reduced Chi2: ", mp.fnorm/len(data), " DOF:", len(data)-len(mpp)

    return mpp, n_gaussian(pars=mpp)(xax), mpperr, chi2


def collapse_gaussfit(cube, xax=None, axis=2, negamp=False, usemoments=True,
                      nsigcut=1.0, mppsigcut=1.0, return_errors=False, **kwargs):
    import time
    std_coll = cube.std(axis=axis)
    std_coll[std_coll == 0] = np.nan  # must eliminate all-zero spectra
    mean_std = median(std_coll[std_coll == std_coll])
    if axis > 0:
        cube = cube.swapaxes(0, axis)
    width_arr = np.zeros(cube.shape[1:]) + np.nan
    amp_arr = np.zeros(cube.shape[1:]) + np.nan
    chi2_arr = np.zeros(cube.shape[1:]) + np.nan
    offset_arr = np.zeros(cube.shape[1:]) + np.nan
    width_err = np.zeros(cube.shape[1:]) + np.nan
    amp_err = np.zeros(cube.shape[1:]) + np.nan
    offset_err = np.zeros(cube.shape[1:]) + np.nan
    if xax is None:
        xax = np.arange(cube.shape[0])
    starttime = time.time()
    print "Cube shape: ", cube.shape
    extremum = np.min if negamp else np.max
    print "Fitting a total of %i spectra with peak signal above %f" %\
          ((np.abs(extremum(cube, axis=0)) > (mean_std*nsigcut)).sum(),
           mean_std*nsigcut)
    for i in xrange(cube.shape[1]):
        t0 = time.time()
        nspec = (np.abs(extremum(cube[:,i,:], axis=0)) > (mean_std*nsigcut)).sum()
        print "Working on row %d with %d spectra to fit" % (i, nspec),
        for j in xrange(cube.shape[2]):
            if np.abs(extremum(cube[:,i,j])) > (mean_std*nsigcut):
                mpp, gfit, mpperr, chi2 = onedgaussfit(xax, cube[:,i,j],
                    err=np.ones(cube.shape[0])*mean_std, negamp=negamp,
                    usemoments=usemoments, **kwargs)
                if np.abs(mpp[1]) > (mpperr[1]*mppsigcut):
                    width_arr[i,j] = mpp[3]
                    offset_arr[i,j] = mpp[2]
                    chi2_arr[i,j] = chi2
                    amp_arr[i,j] = mpp[1]
                    width_err[i,j] = mpperr[3]
                    offset_err[i,j] = mpperr[2]
                    amp_err[i,j] = mpperr[1]
        dt = time.time()-t0
        if nspec > 0:
            print "in %f seconds (average: %f)" % (dt, dt/float(nspec))
        else:
            print "in %f seconds" % (dt)
    print "Total time %f seconds" % (time.time()-starttime)

    if return_errors:
        return width_arr, offset_arr, amp_arr, width_err, offset_err, amp_err, chi2_arr
    else:
        return width_arr, offset_arr, amp_arr, chi2_arr
