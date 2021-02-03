#!/usr/bin/env python3
# -*- coding: utf-8 -*-
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

author = 'Céline Durniak...'

short = 'Repository to test different settings'

setup(name='test_nbsphinx',
      version='0.1.0',
      description=short,
      author=author,
      author_email='  ',
      license=open('LICENSE.txt').read(),
      packages=['QENSmodels'],
      install_requires=['scipy', 'numpy', 'flake8', ],
      zip_safe=False,)
