#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

author = 'Céline Durniak...'

short = 'Repository to test different settings'

setup(name='test_nbsphinx',
      version='0.1.0',
      description=short,
      author=author,
      author_email='  ',
      license=open('LICENSE').read(),
      packages=find_packages(),
      install_requires=['scipy', 'numpy>=1.21', ],
      setup_requires=['flake8'],
      tests_require=['pytest'],
      zip_safe=False,)
