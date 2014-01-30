#!/usr/bin/env python
import sys

if 'develop' in sys.argv:
    # use setuptools for develop, but nothing else
    from setuptools import setup
else:
    from distutils.core import setup

with open('README.rst') as file:
    long_description = file.read()

#execfile('gaussfitter/__version__.py')

setup(name='gaussfitter',
      version='0.1',
      description='gaussfitter, tools for fitting 1d and 2d gaussians',
      long_description=long_description,
      author='Adam Ginsburg',
      author_email='adam.g.ginsburg@gmail.com',
      url='https://github.com/keflavich/gaussfitter',
      packages=['gaussfitter','gaussfitter/mpfit'],
     )
