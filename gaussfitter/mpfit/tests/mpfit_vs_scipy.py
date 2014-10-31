"""
Compare speed and fit quality for a few cases using mpfit and scipy.optimize.leastsq
"""
from agpy.mpfit import mpfit
from agpy.timer import print_timing
from scipy.optimize import leastsq
import scipy.optimize
import numpy as np
import timeit


def gaussian(x,A,dx,w, return_components=False):
    """
    Returns a 1-dimensional gaussian of form
    H+A*numpy.exp(-(x-dx)**2/(2*w**2))
    
    [height,amplitude,center,width]

    return_components does nothing but is required by all fitters
    
    """
    x = np.array(x) # make sure xarr is no longer a spectroscopic axis
    return A*np.exp(-(x-dx)**2/(2.0*w**2))


def error_function_generator(xdata, ydata, error=None, model=gaussian,
        mpfit=False, lmfit=False, sumsquares=False, **kwargs):
    """
    Return a function that generates the errors given input parameters
    """

    if error is None:
        error = np.ones(ydata.shape)

    def error_function(params, **kwargs):
        if lmfit:
            params = [p.value for p in list(params.values())]
        err = (ydata - model(xdata, *params)) / error
        if sumsquares:
            return (err**2).sum()
        elif mpfit:
            return [0,err]
        else:
            return err

    return error_function

def mpfitter(xdata, ydata, params=None, error=None, model=gaussian, quiet=True, **kwargs):
    """
    Find the least-squares fit using mpfit
    """

    errfunc = error_function_generator(xdata,ydata,error=error,model=model,mpfit=True)

    mp = mpfit(errfunc, params, quiet=quiet, **kwargs)

    return mp.params,mp.perror

def lsfitter(xdata, ydata, params=None, error=None, model=gaussian):
    """
    Find the least-squares fit using scipy.optimize.leastsq
    """

    errfunc = error_function_generator(xdata,ydata,error=error,model=model)

    p, cov, infodict, errmsg, success = leastsq(errfunc, params, full_output=1)

    return p, cov.diagonal()**0.5

def annealfitter(xdata, ydata, params=None, error=None, model=gaussian):
    """
    Find the fit using scipy.optimize.anneal
    """

    errfunc = error_function_generator(xdata,ydata,error=error,model=model, sumsquares=True)

    p = scipy.optimize.anneal(errfunc, params, full_output=1)

    return p

def fminfitter(xdata, ydata, params=None, error=None, model=gaussian, disp=False):
    """
    Find the fit using scipy.optimize.fmin
    """

    errfunc = error_function_generator(xdata,ydata,error=error,model=model, sumsquares=True)

    p = scipy.optimize.fmin(errfunc, params, full_output=1, disp=disp)

    return p

def lmfitter(xdata, ydata, params=None, error=None, model=gaussian):
    import lmfit

    errfunc = error_function_generator(xdata,ydata,error=error,model=model,lmfit=True)

    parin = lmfit.Parameters()
    parin.add('amplitude',value=params[0])
    parin.add('shift',value=params[1])
    parin.add('width',value=params[2])

    result = lmfit.minimize(errfunc, parin)

    return [r.value for r in list(parin.values())],[r.stderr for r in list(parin.values())]

if __name__ == "__main__":
    #do some timing
    length = 1000
    xarr = np.linspace(-5,5,length)
    yarr = gaussian(xarr, 0.75, -0.25, 2.2)
    noise = np.random.randn(length) * 0.25
    err = np.ones(length)*0.25

    print(mpfitter(xarr,yarr+noise,[1,0,1],err))
    print(lsfitter(xarr,yarr+noise,[1,0,1],err))
    print(annealfitter(xarr,yarr+noise,[1,0,1],err))
    print(fminfitter(xarr,yarr+noise,[1,0,1],err))
    print(lmfitter(xarr,yarr+noise,[1,0,1],err))

    function_names = ['mpfitter','lsfitter','fminfitter','lmfitter']#,'annealfitter']

    A,dx,s,n = 0.75,-0.25,2.2,0.25

    nfits = 25
    ntries = 7

    print(("%18s" % "nelements")+"".join(["%18s" % fn for fn in function_names]))
    mins = {}
    nels = (2e1,5e1,1e2,2e2,3e2,4e2,5e2,7.5e2,1e3,2.5e3,5e3,1e4,5e4,1e5)
    for nelements in nels:
        min_i = ["%18f" % (min(timeit.Timer("%s(xarr,yarr+noise,[1,0,1],err)" % (fn),
            setup="from mpfit_vs_scipy import %s,gaussian; import numpy as np; xarr=np.linspace(-5,5,%i);\
                    yarr=gaussian(xarr,%f,%f,%f); noise=np.random.randn(%i);\
                    err = np.ones(%i)*%f" % (fn, nelements, A, dx, s, nelements, nelements, n)).repeat(ntries,nfits)))
            for fn in function_names]
        print("%17i:" % (int(nelements)) + "".join(min_i))
        mins[nelements]=min_i

    from pylab import *
    mpmins = array([mins[n][0] for n in nels],dtype='float')
    lsmins = array([mins[n][1] for n in nels],dtype='float')
    fmmins = array([mins[n][2] for n in nels],dtype='float')
    lmfits = array([mins[n][3] for n in nels],dtype='float')
    
    loglog(nels,mpmins,label='mpfit')
    loglog(nels,lsmins,label='leastsq')
    loglog(nels,fmmins,label='fmin')
    loglog(nels,lmfits,label='lmfit')
    xlabel("Number of Elements")
    ylabel("Evaluation Time for %i fits (seconds)" % nfits)
    legend(loc='best')
    savefig("comparison_plot.png")

    clf()
    semilogx(nels,mpmins/lsmins,label='mpfit/leastsq')
    semilogx(nels,fmmins/lsmins,label='fmin/leastsq')
    semilogx(nels,lmfits/lsmins,label='lmfit/leastsq')
    xlabel("Number of Elements")
    ylabel("Ratio to leastsq (which is generally the fastest)")
    legend(loc='best')
    savefig("time_ratio_plot.png")
