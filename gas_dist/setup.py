# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 20:58:45 2020

@author: hsegh
"""

from setuptools import setup

setup(name='gas',
      version='0.1',
      description='A package to preprocess and analyse gas pdg data',
      url='404',
      author_email="hseghetto@gmail.com",
      author='Humberto',
      packages=['gas'],
      install_requires = ["numpy", "matplotlib", "pandas", "tensorflow"],
      license='GPLv3')
