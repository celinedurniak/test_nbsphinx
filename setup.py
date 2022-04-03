#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from setuptools import setup 

author = 'Céline Durniak...'

short = 'Repository to test different settings'

packages = ['QENSmodels']

setup(name='test_nbsphinx',
      version='0.1.0',
      description=short,
      author=author,
      author_email='  ',
      license=open('LICENSE.txt').read(),
      python_requires=">=3.7,<3.11",
      packages=packages,
      install_requires=['scipy', 'numpy', ],
      setup_requires=['flake8'],
      tests_require=['pytest'],
      zip_safe=False,)
