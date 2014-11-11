Gaussfitter
===========

installation (from Linux Terminal): ::
 
   cd /tmp
   git clone https://github.com/keflavich/gaussfitter.git
   cd gaussfitter
   sudo python setup.py install 


This code is taken from agpy, where it has resided for a long time and has had
a long, glorious history.


In short: This is a small toolkit for fitting 2D gaussians.  It makes use of
mpfit.py by Sergei Koposov
(https://code.google.com/p/astrolibpy/source/browse/), and a modified version
of his code is included (by necessity) here.  It is modified primarily to
remove a scipy dependency.

Examples to come!  PRs welcome!

Tested in Python 2.7 and 3.4
