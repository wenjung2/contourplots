# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 20:30:00 2022

@author: sarangbhagwat
"""

from setuptools import setup

setup(
    name='contourplots',
    packages=['contourplots'],
    version='0.2.0',    
    description='A toolkit for generating multi-dimensional plots easily, usefully, and legibly.',
    url='https://github.com/sarangbhagwat/contourplots',
    author='Sarang S. Bhagwat',
    author_email='sarang.bhagwat.git@gmail.com',
    license='MIT',
    # packages=['contourplots'],
    install_requires=['matplotlib==3.5.2',
                      'numpy>=1.23.4',       
                      'imageio>=2.19.3',
                      ],

    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: University of Illinois/NCSA Open Source License',  
        'Operating System :: Microsoft :: Windows',        
        'Programming Language :: Python :: 3.9'
    ],
)
